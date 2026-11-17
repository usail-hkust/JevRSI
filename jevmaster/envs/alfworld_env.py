"""ALFWorld adapter (TextWorld mode, AgentGym AgentEval protocol).

Wraps ``alfworld.agents.environment.alfred_tw_env.AlfredTWEnv`` with
``batch_size=1``, addressing ONE game per episode the way AgentGym's
agentenv-alfworld server does (its ``SingleAlfredTWEnv`` pattern: the game
pool is injected externally -- ``game_files=[one game]`` -> ``init_env`` --
instead of AlfredTWEnv's full-directory scan and fixed-seed shuffle). Legal
actions come from ``info['admissible_commands']``. Requires the ``alfworld``
package, a config YAML, and downloaded data (``alfworld-download`` ->
``$ALFWORLD_DATA``). Imported lazily so the package import never requires it.

Episodes: the AgentEval items (vendored, byte-frozen, at
``data/agenteval/alfworld_test.json`` -- 200 items). ``data_idx`` indexes
AgentGym's concatenated game list -- ``alfworld_mappings_train.json`` (2420
games under ``json_2.1.1/train``) followed by ``alfworld_mappings_test.json``
(200 games under ``json_2.1.1/valid_train``, i.e. items 2420-2619) -- so the
test items resolve to the valid_train games. The official ALFWorld splits are
APPENDED after those 2620 entries by
``scripts/make_alfworld_official_lists.py``
(``alfworld_mappings_official.json``: official train 2620-6172, valid_seen
6173-6312, valid_unseen 6313-6446), so existing indices never move and the new
item lists (``alfworld_train200.json`` / ``alfworld_valid_seen.json`` /
``alfworld_valid_unseen.json``) use the same ``item_id`` schema. ``task`` stays
``"alfworld"`` and ``variation`` is the index into the concatenated list.
Compiling one game per reset costs ~1-3s, the same price AgentGym's server pays
per reset.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Tuple

from jevmaster.envs.agenteval import ITEM_FILES, load_item_indices
from jevmaster.envs.base import EnvAdapter
from jevmaster.envs.registry import register_env
from jevmaster.types import Candidate, Observation, StepResult

_TASK_RE = re.compile(r"your task is to:\s*(.+)", re.IGNORECASE)
# The observation never lists the inventory.  Every command that acts with a held
# object ("move X to Y", "heat X with microwave N", ...) is only admissible while
# X is carried, so the inventory follows exactly from the admissible commands.
_HELD_RE = re.compile(r"^(?:move|put|place|heat|cool|clean) (.+?) (?:to|in|on|with) ",
                      re.IGNORECASE)


def held_objects(admissible) -> List[str]:
    """Objects the agent is carrying, from the admissible commands."""
    out: List[str] = []
    for command in admissible or []:
        match = _HELD_RE.match(str(command))
        if match:
            item = match.group(1).strip()
            if item and item not in out:
                out.append(item)
    return out

# The six ALFWorld task types (the analyzer reports per-type success rates);
# parsed from the gamefile path.
TASK_TYPES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)


def parse_task_type(gamefile: str) -> str:
    """Extract the ALFWorld task type from a gamefile path (or '' if unknown)."""
    if not gamefile:
        return ""
    for name in TASK_TYPES:
        if name in gamefile:
            return name
    return ""


def _make_single_alfred_env(config):
    """Build an AlfredTWEnv with an externally injected game pool (AgentGym's
    ``SingleAlfredTWEnv`` pattern, verbatim): no directory scan, no shuffle --
    the adapter sets ``game_files`` to exactly the game it wants before each
    ``init_env`` call. ``__init__`` deliberately does not call
    ``AlfredTWEnv.__init__``, replicating only the state ``init_env`` needs.
    Defined inline so importing this module never requires alfworld.
    """
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    class SingleAlfredTWEnv(AlfredTWEnv):
        def __init__(self, config):
            print("Initializing AlfredTWEnv...")
            self.config = config
            self.train_eval = "eval"
            self.goal_desc_human_anns_prob = config["env"]["goal_desc_human_anns_prob"]
            self.get_game_logic()
            self.random_seed = 42
            self.game_files = []
            self.num_games = 0

    return SingleAlfredTWEnv(config)


def load_game_pool(root: str, mappings_train: str, mappings_test: str,
                   mappings_official: str = None) -> Tuple[List[str], List[str]]:
    """The concatenated game pool and each entry's split name.

    AgentGym's train mappings (``json_2.1.1/train``) then test mappings
    (``json_2.1.1/valid_train``) come first, keeping the legacy item-list
    indices 0..2619. The official-split mappings (written by
    ``scripts/make_alfworld_official_lists.py``) are appended after them.
    """
    games: List[str] = []
    splits: List[str] = []
    for mapping_file, sub in ((mappings_train, "train"), (mappings_test, "valid_train")):
        with open(mapping_file, "r", encoding="utf-8") as f:
            for m in json.load(f):
                games.append(os.path.join(root, "json_2.1.1", sub,
                                          m["task_type"], m["task_id"], "game.tw-pddl"))
                splits.append(sub)
    # Optional: old checkouts without the file keep serving legacy item lists.
    if mappings_official and os.path.exists(mappings_official):
        with open(mappings_official, "r", encoding="utf-8") as f:
            for m in json.load(f):
                rel = m.get("game")
                if rel:
                    games.append(os.path.join(root, rel))
                else:  # tolerate hand-written records without the path
                    games.append(os.path.join(
                        root, "json_2.1.1", m["split"],
                        m["task_type"], m["task_id"], "game.tw-pddl",
                    ))
                splits.append(str(m.get("split") or ""))
    return games, splits


class ALFWorldEnvAdapter(EnvAdapter):
    name = "alfworld"

    def __init__(
        self,
        config_file: str = None,
        items_file: str = ITEM_FILES["alfworld"],
        mappings_train: str = "data/agenteval/alfworld_mappings_train.json",
        mappings_test: str = "data/agenteval/alfworld_mappings_test.json",
        mappings_official: str = "data/agenteval/alfworld_mappings_official.json",
        data_root: str = None,
        split: str = None,
        **_,
    ):
        if split is not None:
            raise ValueError(
                "alfworld: split-based selection was removed; the AgentEval "
                "item list (env.params.items_file) defines the episode set"
            )
        if not config_file:
            raise ValueError(
                "alfworld requires `env.params.config_file` pointing to an "
                "ALFWorld config YAML (see configs/alfworld.yaml)."
            )
        try:
            import yaml  # noqa: F401  (availability check)
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "alfworld is not installed. Install with `pip install alfworld` "
                "and run `alfworld-download`."
            ) from exc

        with open(config_file, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # AgentGym's concatenated game list: train mappings then test
        # mappings; data_idx (the AgentEval item number) indexes it directly,
        # so items 2420-2619 are the valid_train games. The official splits
        # (scripts/make_alfworld_official_lists.py) follow, so their indices
        # begin at 2620 and existing item lists are untouched.
        root = os.path.expandvars(os.path.expanduser(
            data_root or os.environ.get("ALFWORLD_DATA", "alfworld_data")))
        self._games, self._splits = load_game_pool(
            root, mappings_train, mappings_test, mappings_official)
        self._items = load_item_indices(items_file)
        bad = [i for i in self._items if i >= len(self._games)]
        if bad:
            raise ValueError(
                f"alfworld: item data_idx {bad[:3]}... out of the game list "
                f"(0..{len(self._games) - 1}) -- wrong items_file? "
                f"(official lists need {mappings_official})"
            )
        # The local ALFWORLD_DATA can be partial (239 of the 2420 train games
        # are missing on this pod): fail at init listing the missing ones,
        # not mid-run at the episode that first hits one.
        missing = [p for p in (self._games[i] for i in self._items)
                   if not os.path.exists(p)]
        if missing:
            raise ValueError(
                f"alfworld: {len(missing)} of {len(self._items)} selected game "
                f"files are missing under the local ALFWORLD_DATA ({root}), e.g. "
                f"{missing[0]} -- check $ALFWORLD_DATA or regenerate the item list "
                "over the games that exist locally (scripts/make_agenteval_train.py, "
                "scripts/make_alfworld_official_lists.py)"
            )

        self._tw = _make_single_alfred_env(self._config)
        self._env = None
        self._admissible: List[str] = []
        self._task = "alfworld"
        self._goal = ""
        self._step = 0
        self._won = False
        self._gamefile = ""

    def task_variations(self) -> List[Tuple[str, int]]:
        return [("alfworld", data_idx) for data_idx in self._items]

    def reset(self, task: str, variation: int) -> Observation:
        data_idx = int(variation)
        game_file = self._games[data_idx]
        # One TextWorld env per game: release the previous one before
        # compiling the next (bounds memory/FDs across 200 episodes).
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self._env = None
        self._tw.game_files = [game_file]
        self._tw.num_games = 1
        self._env = self._tw.init_env(batch_size=1)
        obs, info = self._env.reset()
        text = self._first(obs)
        self._admissible = self._first(info.get("admissible_commands", [[]]))
        self._step = 0
        self._won = False
        match = _TASK_RE.search(text)
        self._goal = match.group(1).strip() if match else ""
        # Task type from the gamefile path (per-task-type eval breakdown);
        # falls back to the goal prefix when the gamefile is unavailable.
        gamefile = info.get("extra.gamefile") or info.get("gamefile") or ""
        gamefile = self._first(gamefile) if isinstance(gamefile, (list, tuple)) else gamefile
        self._gamefile = str(gamefile) or game_file
        task_type = parse_task_type(self._gamefile)
        self._task = task_type or (self._goal[:48] if self._goal else "alfworld")
        return Observation(
            task=self._task,
            goal=self._goal,
            text=text,
            inventory="",
            step=0,
            extra={"data_idx": data_idx, "split": self._splits[data_idx],
                   "gamefile": self._gamefile, "task_type": task_type},
        )

    @staticmethod
    def _first(value):
        if isinstance(value, (list, tuple)) and value:
            return value[0]
        return value

    def candidates(self) -> List[Candidate]:
        seen = set()
        out: List[Candidate] = []
        for a in self._admissible:
            a = str(a)
            if a not in seen:
                seen.add(a)
                out.append(Candidate(a))
        return out

    def step(self, action: str) -> StepResult:
        obs, scores, dones, info = self._env.step([action])
        self._step += 1
        text = self._first(obs)
        self._admissible = self._first(info.get("admissible_commands", [[]]))
        reward = float(self._first(scores) or 0.0)
        done = bool(self._first(dones))
        won = bool(self._first(info.get("won", [False])))
        self._won = self._won or won
        return StepResult(
            observation=Observation(
                task=self._task,
                goal=self._goal,
                text=text,
                inventory=self._inventory(),
                step=self._step,
                extra={},
            ),
            reward=reward,
            done=done,
            info={"won": won},
        )

    def _inventory(self) -> str:
        """What the agent carries, stated for the observation (see held_objects)."""
        held = held_objects(self._admissible)
        return "In your inventory, you see: " + ", ".join(held) if held else ""

    def score(self) -> float:
        return 1.0 if self._won else 0.0

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
            self._env = None


@register_env("alfworld")
def _make_alfworld(**kwargs) -> ALFWorldEnvAdapter:
    return ALFWorldEnvAdapter(**kwargs)
