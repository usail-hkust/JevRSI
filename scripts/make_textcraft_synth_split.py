#!/usr/bin/env python
"""Build the frozen TextCraft-Synth official split (train / ID / OOD).

TextCraft-Synth is platoon's synthetic crafting benchmark
(``platoon-textcraft`` 0.4.0, MIT; RAO paper, arXiv 2605.06639). It ships
2,522 train and 632 validation tasks, each tagged with ``misc.max_depth``
(the crafting-tree depth of its single target). This script slices the
published train file into the recommended official profile:

    split  recipe depth  count  selection
    train  2-4           200    first slice of the depth-2/3/4 pool
    id     2-4           150    next slice of the same pool (new instances)
    ood    5-6           150    depth-5/6 pool (deeper trees, unseen goals)

Selection is deterministic: each depth bucket is sorted by its numeric task
id, then the buckets are round-robin interleaved (2,3,4,2,3,4,...), so every
prefix of a split covers the whole depth range instead of one small group.
``train`` and ``id`` are disjoint slices of the same pool; ``ood`` comes from
the deeper bucket. All three are disjoint sets of task instances.

The world (1,452 synthetic items, 1,402 recipes) is generated once with the
vendored generator (seed 42, 25 items per domain/tier) and written compactly
as ``data/textcraft_synth/recipes.json`` -- the same ``textcraft-recipes/1``
format platoon's Harbor export uses. Task files are not committed (25 MB);
pass ``--download`` or ``--tasks`` to point at the published jsonl.

Every selected task is validated before writing: its targets and starting
inventory must exist in the world, and its gold trajectory must replay to a
score of 1.0 through the vendored benchmark ``Game`` rules.

Usage
-----
    # generate recipes + manifests from the published train file
    /opt/python/bin/python scripts/make_textcraft_synth_split.py --download

    # use a local copy of textcraft_synth_train.jsonl
    /opt/python/bin/python scripts/make_textcraft_synth_split.py \
        --tasks /path/to/textcraft_synth_train.jsonl

    # validate the frozen files without the source jsonl
    /opt/python/bin/python scripts/make_textcraft_synth_split.py --check

Outputs (tracked, benchmark definitions):
    data/textcraft_synth/recipes.json
    data/textcraft_synth/splits/textcraft_synth_train.json
    data/textcraft_synth/splits/textcraft_synth_id.json
    data/textcraft_synth/splits/textcraft_synth_ood.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jevmaster.envs.textcraft_synth import Game, RecipeBook  # noqa: E402
from jevmaster.envs.textcraft_synth.game import (  # noqa: E402
    RECIPES_FORMAT, TASK_FORMAT,
)
from jevmaster.envs.textcraft_synth.synth_recipe_generator import (  # noqa: E402
    SynthRecipeDatabase,
)

SOURCE_URL = (
    "https://raw.githubusercontent.com/ApGa/platoon/apga/rao-snapshot/"
    "plugins/textcraft/platoon/textcraft/textcraft_synth_train.jsonl"
)
SOURCE_SHA256 = "685823fbca90aa89e061e61fef58d0122def4221eed14cf2c8c730341363a00f"
SOURCE_NAME = "textcraft_synth_train.jsonl"

OUT_DIR = REPO_ROOT / "data/textcraft_synth"
SOURCE_DIR = OUT_DIR / "source"
SPLIT_DIR = OUT_DIR / "splits"
RECIPES_FILE = OUT_DIR / "recipes.json"

RECIPE_SEED = 42
ITEMS_PER_DOMAIN_TIER = 25

# The delivered official profile: split -> (depths, count).
SPLIT_DEPTHS = {"train": (2, 4), "id": (2, 4), "ood": (5, 6)}
SPLIT_COUNTS = {"train": 200, "id": 150, "ood": 150}
# Which pool feeds which split. train/id share the 2-4 pool; ood is deeper.
SPLIT_POOL = {"train": "shallow", "id": "shallow", "ood": "deep"}


# --------------------------------------------------------------------- source

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f">> downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=600) as response, open(dest, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    return dest


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; pass --tasks or --download to fetch the "
            f"published {SOURCE_NAME}"
        )
    digest = _sha256(path)
    if digest != SOURCE_SHA256:
        raise ValueError(
            f"{path}: sha256 {digest} does not match the pinned "
            f"{SOURCE_SHA256} (wrong or modified source file)"
        )
    tasks = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            misc = task.get("misc") or {}
            for key in ("target_items", "initial_inventory", "gold_trajectory", "max_depth"):
                if key not in misc:
                    raise ValueError(f"{path}:{line_no}: missing misc.{key}")
            tasks.append(task)
    return tasks


# ------------------------------------------------------------------ the world

def build_recipe_db() -> SynthRecipeDatabase:
    db = SynthRecipeDatabase()
    state = random.getstate()
    try:
        db.generate_all_recipes(seed=RECIPE_SEED,
                                items_per_domain_tier=ITEMS_PER_DOMAIN_TIER)
    finally:
        random.setstate(state)
    return db


def recipes_document(db: SynthRecipeDatabase, world: str) -> Dict[str, Any]:
    """The ``textcraft-recipes/1`` document (platoon's Harbor export format)."""
    items = set(db.recipes) | set(db.base_items) | set(db.all_items)
    tags = set(getattr(db, "all_tags", ()))
    recipes: Dict[str, List[Dict[str, Any]]] = {}
    for item in sorted(db.recipes):
        entries = []
        for recipe in db.get_recipes_for_item(item):
            entries.append({
                "result_count": recipe.result_count,
                "ingredients": [[name, count] for name, count in recipe.ingredients.items()],
            })
            tags |= {name[4:] for name in recipe.ingredients if name.startswith("tag:")}
        recipes[item] = entries
    depths = {item: depth for item in sorted(items)
              if (depth := db.get_crafting_depth(item))}
    return {
        "format": RECIPES_FORMAT,
        "name": world,
        "base_items": sorted(db.base_items),
        "depths": depths,
        "tags": {tag: list(db.get_items_for_tag(tag)) for tag in sorted(tags)},
        "recipes": recipes,
    }


def encode_recipes(document: Dict[str, Any]) -> bytes:
    """Compact JSON, one line per item's recipes (platoon's encoder)."""
    def dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    lines = ["{"]
    for key in ("format", "name", "base_items", "depths", "tags"):
        lines.append(f"{dump(key)}:{dump(document[key])},")
    lines.append('"recipes":{')
    entries = list(document["recipes"].items())
    for number, (item, recipes) in enumerate(entries):
        lines.append(f"{dump(item)}:{dump(recipes)}"
                     + ("," if number < len(entries) - 1 else ""))
    lines.append("}}")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_recipes(path: Path) -> RecipeBook:
    db = build_recipe_db()
    document = recipes_document(db, "textcraft-synth")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_recipes(document))
    book = RecipeBook.load(path)
    # The exported world must answer every lookup exactly as the generator.
    names = set(db.recipes) | set(db.base_items) | set(db.all_items)
    for item in sorted(names):
        if (book.can_craft(item) != db.can_craft(item)
                or book.is_base_item(item) != db.is_base_item(item)
                or book.get_crafting_depth(item) != db.get_crafting_depth(item)):
            raise ValueError(f"recipes.json describes {item!r} unlike the generator")
        mine = [(r.result_count, list(r.ingredients.items()))
                for r in book.get_recipes_for_item(item)]
        theirs = [(r.result_count, list(r.ingredients.items()))
                  for r in db.get_recipes_for_item(item)]
        if mine != theirs:
            raise ValueError(f"recipes.json has different recipes for {item!r}")
    print(f">> wrote {path.relative_to(REPO_ROOT)} "
          f"({path.stat().st_size / 1024:.0f} KiB, {len(book.recipes)} items)")
    return book


# ------------------------------------------------------------------ selection

def _task_number(task: Dict[str, Any]) -> int:
    return int(str(task["id"]).rsplit(".", 1)[1])


def interleave_by_depth(tasks: Iterable[Dict[str, Any]], depths: Sequence[int]) -> List[Dict[str, Any]]:
    """Round-robin over depth buckets, each sorted by its numeric task id."""
    buckets: Dict[int, deque] = {}
    for depth in depths:
        members = sorted((t for t in tasks if int(t["misc"]["max_depth"]) == depth),
                         key=_task_number)
        buckets[depth] = deque(members)
    ordered: List[Dict[str, Any]] = []
    while any(buckets.values()):
        for depth in depths:
            if buckets[depth]:
                ordered.append(buckets[depth].popleft())
    return ordered


def select_tasks(tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    shallow = interleave_by_depth(tasks, range(SPLIT_DEPTHS["train"][0],
                                               SPLIT_DEPTHS["train"][1] + 1))
    deep = interleave_by_depth(tasks, range(SPLIT_DEPTHS["ood"][0],
                                            SPLIT_DEPTHS["ood"][1] + 1))
    pools = {"shallow": shallow, "deep": deep}
    cursor = {"shallow": 0, "deep": 0}
    chosen: Dict[str, List[Dict[str, Any]]] = {}
    for split in ("train", "id", "ood"):
        pool = pools[SPLIT_POOL[split]]
        count = SPLIT_COUNTS[split]
        start = cursor[SPLIT_POOL[split]]
        picked = pool[start:start + count]
        if len(picked) != count:
            raise ValueError(
                f"{split}: pool {SPLIT_POOL[split]} has only "
                f"{len(pool) - start} tasks left, need {count}")
        cursor[SPLIT_POOL[split]] = start + count
        chosen[split] = picked
    return chosen


# ----------------------------------------------------------------- validation

def _recipe_for(book: RecipeBook, item: str, ingredients: Dict[str, int],
                produced: int):
    """The recipe + batch count the gold step used, or None."""
    for recipe in book.get_recipes_for_item(item):
        if set(recipe.ingredients) != set(ingredients):
            continue
        batches = None
        ok = True
        for name, per_batch in recipe.ingredients.items():
            if ingredients[name] % per_batch:
                ok = False
                break
            n = ingredients[name] // per_batch
            if batches is None:
                batches = n
            elif batches != n:
                ok = False
                break
        if ok and batches and produced == batches * recipe.result_count:
            return recipe, batches
    return None


def record_for(split: str, index: int, task: Dict[str, Any], book: RecipeBook) -> Dict[str, Any]:
    misc = task["misc"]
    targets = dict(misc["target_items"])
    initial = dict(misc["initial_inventory"])
    game = Game(book, {
        "format": TASK_FORMAT,
        "name": str(task["id"]),
        "goal": task["goal"],
        "target_items": targets,
        "initial_inventory": initial,
        "recipes_sha256": book.sha256,
    })
    for step in misc["gold_trajectory"]:
        item, _requested = step["target"]
        produced = int(step["result_count"])
        ingredients = dict(step["ingredients"])
        if _recipe_for(book, item, ingredients, produced) is None:
            raise ValueError(f"{task['id']}: gold step {item!r} matches no recipe")
        result = game.craft(ingredients, (item, produced))
        if not result.startswith("Successfully crafted"):
            raise ValueError(f"{task['id']}: gold step failed: {result}")
    score, _ = game.evaluate("gold trajectory")
    if score != 1.0:
        raise ValueError(f"{task['id']}: gold trajectory does not reach the target")
    for item in targets:
        if not book.can_craft(item):
            raise ValueError(f"{task['id']}: target {item!r} is not craftable")

    record = {
        "item_id": f"textcraft_synth_{split}_{index:05d}",
        "split": split,
        "task": "craft",
        "goal": task["goal"],
        "target_items": targets,
        "initial_inventory": initial,
        "max_steps": int(task.get("max_steps") or 75),
        "max_depth": int(misc["max_depth"]),
        "difficulty": misc.get("difficulty"),
        "source": "platoon-textcraft 0.4.0",
        "source_id": task["id"],
    }
    return record


def build_manifests(tasks: List[Dict[str, Any]], book: RecipeBook) -> Dict[str, List[Dict[str, Any]]]:
    chosen = select_tasks(tasks)
    manifests = {}
    for split, picked in chosen.items():
        manifests[split] = [record_for(split, i, task, book)
                            for i, task in enumerate(picked)]
        depths = [r["max_depth"] for r in manifests[split]]
        print(f">> {split}: {len(manifests[split])} tasks, depth "
              f"{min(depths)}-{max(depths)}")
    return manifests


def write_manifests(manifests: Dict[str, List[Dict[str, Any]]]) -> None:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for split, records in manifests.items():
        path = SPLIT_DIR / f"textcraft_synth_{split}.json"
        path.write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f">> wrote {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------- check

def check(book: RecipeBook) -> int:
    """Validate the frozen manifests; returns the number of problems."""
    problems = 0
    seen_ids = set()
    seen_sources = set()
    for split, count in SPLIT_COUNTS.items():
        path = SPLIT_DIR / f"textcraft_synth_{split}.json"
        if not path.exists():
            print(f"!! missing {path}")
            problems += 1
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        if len(records) != count:
            print(f"!! {path.name}: {len(records)} tasks, expected {count}")
            problems += 1
        low, high = SPLIT_DEPTHS[split]
        for record in records:
            if not (low <= int(record["max_depth"]) <= high):
                print(f"!! {record['item_id']}: depth {record['max_depth']} "
                      f"outside {low}-{high}")
                problems += 1
            if record["item_id"] in seen_ids:
                print(f"!! duplicate item_id {record['item_id']}")
                problems += 1
            seen_ids.add(record["item_id"])
            if record["source_id"] in seen_sources:
                print(f"!! {record['item_id']}: source task {record['source_id']!r} "
                      f"appears in more than one split")
                problems += 1
            seen_sources.add(record["source_id"])
            for item in record["target_items"]:
                if not book.can_craft(item):
                    print(f"!! {record['item_id']}: target {item!r} is not craftable")
                    problems += 1
            for item in record["initial_inventory"]:
                if not (book.is_base_item(item) or book.can_craft(item)):
                    print(f"!! {record['item_id']}: inventory item {item!r} unknown")
                    problems += 1
            if int(record.get("max_steps") or 0) <= 0:
                print(f"!! {record['item_id']}: max_steps must be positive")
                problems += 1
        print(f">> {split}: {len(records)} tasks checked "
              f"(depth {min(r['max_depth'] for r in records)}-"
              f"{max(r['max_depth'] for r in records)})")
    print(f">> {'OK' if not problems else str(problems) + ' problem(s)'}")
    return problems


# ----------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, default=SOURCE_DIR / SOURCE_NAME,
                        help=f"published {SOURCE_NAME} (default: %(default)s)")
    parser.add_argument("--download", action="store_true",
                        help="download the published train file when --tasks is missing")
    parser.add_argument("--check", action="store_true",
                        help="validate the frozen manifests / recipes and exit")
    args = parser.parse_args()

    if args.check:
        if not RECIPES_FILE.exists():
            print(f"!! missing {RECIPES_FILE}")
            return 1
        return 1 if check(RecipeBook.load(RECIPES_FILE)) else 0

    path = args.tasks
    if not path.exists() and args.download:
        path = download_source(path)
    tasks = load_tasks(path)
    print(f">> loaded {len(tasks)} tasks from {path}")

    book = write_recipes(RECIPES_FILE)
    manifests = build_manifests(tasks, book)
    write_manifests(manifests)
    return 1 if check(book) else 0


if __name__ == "__main__":
    raise SystemExit(main())
