"""TextCraft-Synth adapter (platoon-textcraft official split).

TextCraft-Synth is platoon's synthetic variant of TextCraft (RAO paper,
arXiv 2605.06639; ``platoon-textcraft`` 0.4.0, MIT): a procedurally generated
crafting world with deeper recipe trees than Minecraft. The world is frozen
as ``data/textcraft_synth/recipes.json`` (seed 42, 25 items per domain/tier,
1,452 items), and the official split manifests under
``data/textcraft_synth/splits/`` are 200 train / 150 ID (recipe depth 2-4,
disjoint instances) / 150 OOD (depth 5-6), built by
``scripts/make_textcraft_synth_split.py``.

The vendored benchmark rules (:class:`jevmaster.envs.textcraft_synth.Game`)
are kept exactly: an episode starts from the task's inventory, the tools are
``craft`` / ``get_info`` / ``view_inventory`` / ``finish``, and success is 1.0
only after ``finish`` with a non-empty message and every target crafted on
top of the starting inventory (items already present at reset do not count).

Actions are the benchmark's tools in the same string form the original
TextCraft adapter uses:

- ``craft N <item> using N1 <ing1>, N2 <ing2>, ...`` -- N is the total to
  produce and must be divisible by the recipe's result count; ingredient
  counts are the recipe's counts scaled by ``N / result_count``;
- ``get_info <item>[, <item>...]`` / ``inventory`` -- read-only queries;
- ``finish <message>`` -- end the episode and grade it.

``Observation.text`` carries the recipe book for the dependency closure of
the targets (the benchmark's ``get_info`` is therefore optional), and
``Observation.inventory`` exposes the live inventory in the env's bracket
format. Candidates are ``inventory`` plus every recipe the current inventory
can execute once, and ``finish`` once the target counts have been crafted.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jevmaster.envs.base import EnvAdapter
from jevmaster.envs.registry import register_env
from jevmaster.envs.splits import load_items
from jevmaster.envs.textcraft_synth import Game, RecipeBook
from jevmaster.envs.textcraft_synth.game import TASK_FORMAT
from jevmaster.types import Candidate, Observation, StepResult

logger = logging.getLogger(__name__)

TASK = "craft"

DEFAULT_ITEMS_FILE = "data/textcraft_synth/splits/textcraft_synth_id.json"
DEFAULT_RECIPES_FILE = "data/textcraft_synth/recipes.json"

_CRAFT_RE = re.compile(r"^craft\s+(\d+)\s+(\S+)\s+using\s+(.+)$", re.IGNORECASE)
_ING_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s*$")
_INFO_RE = re.compile(r"^(?:get_info|info)\s+(.+)$", re.IGNORECASE)
_FINISH_RE = re.compile(r"^finish(?:\s+(.*))?$", re.IGNORECASE | re.DOTALL)

CRAFTED = "Successfully crafted"
FINISH_ACTION = "finish crafted all targets"


def _inventory_str(inventory: Dict[str, int]) -> str:
    return " ".join(f"[{item}] ({count})" for item, count in sorted(inventory.items()))


class TextCraftSynthEnvAdapter(EnvAdapter):
    name = "textcraft_synth"

    def __init__(
        self,
        items_file: str = DEFAULT_ITEMS_FILE,
        recipes_file: str = DEFAULT_RECIPES_FILE,
        split: str = None,
        **_,
    ):
        path = Path(items_file)
        if not path.exists():
            raise FileNotFoundError(
                f"textcraft_synth: items file not found: {path} (run from the "
                "repo root; the official manifests live under "
                "data/textcraft_synth/splits/)"
            )
        self._items = load_items(path, split=split)
        self._book = RecipeBook.load(recipes_file)
        for index, record in enumerate(self._items):
            self._validate_record(index, record)

        self._game: Game | None = None
        self._record: Dict[str, Any] = {}
        self._book_text = ""
        self._goal = ""
        self._step = 0
        self._max_steps = 75
        self._score = 0.0

    # ---- episode selection ------------------------------------------------

    def task_variations(self) -> List[Tuple[str, int]]:
        return [(TASK, index) for index in range(len(self._items))]

    # ---- env lifecycle ----------------------------------------------------

    def reset(self, task: str, variation: int) -> Observation:
        if task != TASK:
            raise ValueError(f"textcraft_synth: unknown task {task!r} (only {TASK!r})")
        record = self._items[int(variation)]
        self._record = record
        self._goal = str(record["goal"])
        self._step = 0
        self._max_steps = int(record.get("max_steps") or 75)
        self._score = 0.0
        self._game = Game(self._book, {
            "format": TASK_FORMAT,
            "name": str(record.get("source_id") or record["item_id"]),
            "goal": self._goal,
            "target_items": dict(record["target_items"]),
            "initial_inventory": dict(record["initial_inventory"]),
            "recipes_sha256": self._book.sha256,
        })
        self._book_text = self._recipe_book_text(record["target_items"])
        return self._observation("")

    def step(self, action: str) -> StepResult:
        if self._game is None:
            raise RuntimeError("textcraft_synth: reset() before step()")
        if self._game.finished:
            return StepResult(observation=self._observation(""), reward=self._score,
                              done=True, info=self._info())
        self._step += 1
        feedback = self._apply(str(action or "").strip())
        done = self._game.finished or self._step >= self._max_steps
        if done:
            self._score = float(self._game.score) if self._game.finished else 0.0
        return StepResult(
            observation=self._observation(feedback),
            reward=self._score,
            done=done,
            info=self._info(),
        )

    def score(self) -> float:
        return self._score

    def close(self) -> None:
        return None

    # ---- candidates -------------------------------------------------------

    def candidates(self) -> List[Candidate]:
        out = [Candidate("inventory")]
        inventory = self._game.inventory if self._game else {}
        seen = set()
        for item, depth in sorted(self._book.depths.items(), key=lambda kv: (kv[1], kv[0])):
            for recipe in self._book.get_recipes_for_item(item):
                if any(inventory.get(name, 0) < count
                       for name, count in recipe.ingredients.items()):
                    continue
                action = "craft {} {} using {}".format(
                    recipe.result_count, item,
                    ", ".join(f"{count} {name}" for name, count in recipe.ingredients.items()),
                )
                if action not in seen:
                    seen.add(action)
                    out.append(Candidate(action))
        if self._targets_met():
            out.append(Candidate(FINISH_ACTION))
        return out

    # ---- action execution -------------------------------------------------

    def _apply(self, action: str) -> str:
        if action in ("inventory", "view_inventory"):
            return json.dumps(dict(self._game.inventory), sort_keys=True)
        match = _INFO_RE.match(action)
        if match:
            items = [item for item in re.split(r"[,\s]+", match.group(1).strip()) if item]
            return json.dumps(self._game.get_info(items), ensure_ascii=False)
        match = _FINISH_RE.match(action)
        if match:
            return self._game.call("finish", {"message": (match.group(1) or "").strip()})
        match = _CRAFT_RE.match(action)
        if match:
            count, item, ingredients_text = int(match.group(1)), match.group(2), match.group(3)
            ingredients: Dict[str, int] = {}
            for part in ingredients_text.split(","):
                ing = _ING_RE.match(part)
                if not ing:
                    return f"Could not execute {action}"
                name, amount = ing.group(2), int(ing.group(1))
                ingredients[name] = ingredients.get(name, 0) + amount
            return self._game.call("craft", {"ingredients": ingredients,
                                             "target": [item, count]})
        return f"Could not execute {action}"

    # ---- helpers ----------------------------------------------------------

    def _targets_met(self) -> bool:
        if self._game is None:
            return False
        for item, required in self._game.target_items.items():
            gained = (self._game.inventory.get(item, 0)
                      - self._game.initial_inventory.get(item, 0))
            if gained < required:
                return False
        return True

    def _info(self) -> Dict[str, Any]:
        return {
            "item_id": self._record.get("item_id"),
            "target_items": dict(self._record.get("target_items") or {}),
            "max_depth": self._record.get("max_depth"),
            "success": bool(self._game.finished and self._game.score >= 1.0),
        }

    def _observation(self, feedback: str) -> Observation:
        text = self._book_text
        if feedback:
            text = f"{text}\n\n{feedback}" if text else feedback
        return Observation(
            task=TASK,
            goal=self._goal,
            text=text,
            inventory=_inventory_str(self._game.inventory),
            step=self._step,
            extra=self._info(),
        )

    def _recipe_book_text(self, targets: Dict[str, int]) -> str:
        closure = self._closure(targets)
        lines = []
        for item in sorted(closure, key=lambda name: (self._book.get_crafting_depth(name), name)):
            for recipe in self._book.get_recipes_for_item(item):
                lines.append("craft {} {} using {}".format(
                    recipe.result_count, item,
                    ", ".join(f"{count} {name}" for name, count in recipe.ingredients.items()),
                ))
        return "Crafting commands:\n" + ("\n".join(lines) or "(none)")

    def _closure(self, targets: Dict[str, int]) -> set:
        seen = set()
        stack = list(targets)
        while stack:
            item = stack.pop()
            if item in seen:
                continue
            recipes = self._book.get_recipes_for_item(item)
            if not recipes:
                continue
            seen.add(item)
            for recipe in recipes:
                stack.extend(recipe.ingredients)
        return seen

    def _validate_record(self, index: int, record: Dict[str, Any]) -> None:
        item_id = record.get("item_id") or f"<record {index}>"
        targets = record.get("target_items")
        inventory = record.get("initial_inventory")
        if not isinstance(targets, dict) or not targets:
            raise ValueError(f"textcraft_synth: {item_id}: missing target_items")
        if not isinstance(inventory, dict):
            raise ValueError(f"textcraft_synth: {item_id}: missing initial_inventory")
        for item in targets:
            if not self._book.can_craft(item):
                raise ValueError(f"textcraft_synth: {item_id}: target {item!r} is not craftable")
        for item in inventory:
            if not (self._book.is_base_item(item) or self._book.can_craft(item)):
                raise ValueError(f"textcraft_synth: {item_id}: inventory item {item!r} unknown")
        if int(record.get("max_steps") or 0) <= 0:
            raise ValueError(f"textcraft_synth: {item_id}: max_steps must be positive")


@register_env("textcraft_synth")
def _make_textcraft_synth(**kwargs) -> TextCraftSynthEnvAdapter:
    return TextCraftSynthEnvAdapter(**kwargs)
