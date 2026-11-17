#!/usr/bin/env python
"""Build the frozen official-split ALFWorld lists (train 200 / valid_seen / valid_unseen).

The AgentEval protocol indexes ALFWorld's game pool through
``data/agenteval/alfworld_mappings_{train,test}.json``: train mappings (2420
games under ``json_2.1.1/train``) followed by test mappings (200 games under
``json_2.1.1/valid_train``). This script APPENDS the official ALFWorld splits
after those 2620 entries, so every existing item list keeps its exact indices:

    0 .. 2619          legacy AgentEval mappings (unchanged)
    2620 .. 6172       official train       (3553 games)
    6173 .. 6312       official valid_seen  (140 games, ID)
    6313 .. 6446       official valid_unseen (134 games, OOD)

It writes:

    data/agenteval/alfworld_mappings_official.json  (3827 records, appended order)
    data/agenteval/alfworld_train200.json           (200 train items, stratified)
    data/agenteval/alfworld_valid_seen.json         (140 items, ID)
    data/agenteval/alfworld_valid_unseen.json       (134 items, OOD)

Game selection mirrors ``alfworld.agents.environment.alfred_tw_env
.AlfredTWEnv.collect_game_files``: skip paths containing ``movable`` or
``Sliced``, require ``game.tw-pddl`` and a truthy ``solvable`` flag, and keep
only the task types enabled in the ALFWorld base config (default 1..6).

The train list is the first 200 games of a round-robin over the six task
types (sorted order within each type), so the warm-up covers all types evenly.
The selected order is frozen into the item file: every run of the protocol
reads the same 200 games. Official counts are asserted (3553 / 140 / 134); a
mismatch means the mounted ``$ALFWORLD_DATA`` is partial or filtered
differently. Scans are cached under ``/tmp/alfworld_official_scan`` (the
official data is immutable); ``--refresh`` rescans, and ``--verify-solvable``
reads the selected games' ``solvable`` flags (slow on the mount).

    python scripts/make_alfworld_official_lists.py            # write the files
    python scripts/make_alfworld_official_lists.py --check    # verify, no writes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import OrderedDict, deque
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data/agenteval"
BASE_CONFIG = REPO_ROOT / "configs/alfworld_base.yaml"

# Canonical task types (ALFWorld base config ids 1-6). Task type 7
# (pick_and_place_with_movable_recep) is excluded by the default config and is
# also skipped by the 'movable' path filter.
TASK_TYPES = OrderedDict(
    [
        (1, "pick_and_place_simple"),
        (2, "look_at_obj_in_light"),
        (3, "pick_clean_then_place_in_recep"),
        (4, "pick_heat_then_place_in_recep"),
        (5, "pick_cool_then_place_in_recep"),
        (6, "pick_two_obj_and_place"),
    ]
)

SPLITS = OrderedDict(
    [
        ("train", "json_2.1.1/train"),
        ("valid_seen", "json_2.1.1/valid_seen"),
        ("valid_unseen", "json_2.1.1/valid_unseen"),
    ]
)
EXPECTED = {"train": 3553, "valid_seen": 140, "valid_unseen": 134}
TRAIN_ITEMS = 200

MAPPINGS_TRAIN = DATA_DIR / "alfworld_mappings_train.json"
MAPPINGS_TEST = DATA_DIR / "alfworld_mappings_test.json"
MAPPINGS_OFFICIAL = DATA_DIR / "alfworld_mappings_official.json"
ITEM_FILES = {
    "train": DATA_DIR / "alfworld_train200.json",
    "valid_seen": DATA_DIR / "alfworld_valid_seen.json",
    "valid_unseen": DATA_DIR / "alfworld_valid_unseen.json",
}


def data_root(explicit: str | None = None) -> Path:
    """Resolve $ALFWORLD_DATA (or --data-root), matching the adapter."""
    root = explicit or os.environ.get("ALFWORLD_DATA") or str(REPO_ROOT / "alfworld_data")
    return Path(os.path.expandvars(os.path.expanduser(str(root))))


def enabled_types(base_config: Path = BASE_CONFIG) -> list[str]:
    """Canonical task-type names enabled by the ALFWorld base config."""
    if not base_config.exists():
        return [TASK_TYPES[i] for i in TASK_TYPES]
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    ids = (cfg.get("env") or {}).get("task_types") or list(TASK_TYPES)
    return [TASK_TYPES[int(i)] for i in ids if int(i) in TASK_TYPES]


def canonical_type(task_dir: str, wanted: list[str]) -> str | None:
    """The canonical six-way type a task directory belongs to, or None."""
    for name in wanted:
        if task_dir == name or task_dir.startswith(name + "-"):
            return name
    return None


def scan_split(root: Path, split: str, wanted: list[str],
               refresh: bool = False) -> list[dict]:
    """Official-filter game records for one split, sorted by relative path.

    Same filters as ``AlfredTWEnv.collect_game_files``: skip ``movable`` /
    ``Sliced`` paths, require ``game.tw-pddl``, keep only enabled task types.
    The ``solvable`` flag is checked separately (``--verify-solvable``); a full
    read of thousands of 70KB JSON files is prohibitively slow on the mounted
    dataset, and its counts already match the published split sizes.

    Scans are cached under ``/tmp/alfworld_official_scan/<split>.json`` so a
    slow mounted filesystem is walked once; pass ``--refresh`` to rescan.
    """
    split_root = root / SPLITS[split]
    if not split_root.is_dir():
        raise SystemExit(
            f"missing split directory: {split_root}\n"
            f"export ALFWORLD_DATA to the downloaded alfworld_data "
            f"(currently {root})"
        )
    cache_dir = Path(tempfile.gettempdir()) / "alfworld_official_scan"
    digest = hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:12]
    cache = cache_dir / f"{split}-{digest}.json"
    if cache.exists() and not refresh:
        records = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  {split}: {len(records)} games (cached {cache})", flush=True)
        return records
    records: list[dict] = []
    seen = 0
    for dirpath, _dirs, files in os.walk(split_root):
        if "game.tw-pddl" not in files:
            continue
        if "movable" in dirpath or "Sliced" in dirpath:
            continue
        seen += 1
        if seen % 500 == 0:
            print(f"  {split}: scanned {seen} games...", flush=True)
        game = Path(dirpath) / "game.tw-pddl"
        rel = game.relative_to(root).as_posix()
        parts = game.relative_to(split_root).parts  # task_dir / trial / game
        task_dir = parts[0] if parts else ""
        task_type = canonical_type(task_dir, wanted)
        if task_type is None:
            continue
        trial = parts[1] if len(parts) >= 3 else ""
        records.append(
            {
                "split": split,
                "task_type": task_dir,
                "canonical_type": task_type,
                "task_id": trial,
                "game": rel,
            }
        )
    records.sort(key=lambda r: r["game"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    print(f"  {split}: {len(records)} games", flush=True)
    return records


def is_solvable(root: Path, record: dict) -> bool:
    """The game file exists and carries a truthy ``solvable`` flag."""
    path = root / record["game"]
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(data.get("solvable"))


def stratified_train(root: Path, records: list[dict], n: int,
                     verify_solvable: bool = False) -> list[dict]:
    """Round-robin over task types (sorted order within each), first ``n`` games.

    Deterministic: the same ``$ALFWORLD_DATA`` always yields the same list, and
    the resulting item file is committed. With ``verify_solvable`` unplayable
    (missing / ``solvable`` false) games are skipped and replaced by the next
    game of the same type.
    """
    queues: dict[str, deque] = {}
    for rec in records:
        queues.setdefault(rec["canonical_type"], deque()).append(rec)
    order = [t for t in TASK_TYPES.values() if t in queues]
    selected: list[dict] = []
    while len(selected) < n:
        progressed = False
        for task_type in order:
            queue = queues.get(task_type)
            while queue:
                rec = queue.popleft()
                if verify_solvable and not is_solvable(root, rec):
                    continue
                selected.append(rec)
                progressed = True
                break
            if len(selected) >= n:
                break
        if not progressed:
            break
    if len(selected) != n:
        raise SystemExit(
            f"only {len(selected)} playable games for the stratified train list "
            f"(wanted {n}); check $ALFWORLD_DATA"
        )
    return selected


def legacy_index_start() -> int:
    """Index of the first appended official record (the legacy list length)."""
    total = 0
    for path in (MAPPINGS_TRAIN, MAPPINGS_TEST):
        if not path.exists():
            raise SystemExit(f"missing legacy mappings file: {path}")
        total += len(json.loads(path.read_text(encoding="utf-8")))
    return total


def build(root: Path, wanted: list[str], *, refresh: bool = False,
          verify_solvable: bool = False) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """(official games per split, selected item records per list)."""
    by_split = {split: scan_split(root, split, wanted, refresh=refresh) for split in SPLITS}
    for split, expected in EXPECTED.items():
        found = len(by_split[split])
        if found != expected:
            raise SystemExit(
                f"{split}: found {found} games, expected {expected} -- "
                f"$ALFWORLD_DATA ({root}) is partial or filtered differently"
            )
    train200 = stratified_train(root, by_split["train"], TRAIN_ITEMS,
                                verify_solvable=verify_solvable)
    selected = {
        "train": train200,
        "valid_seen": by_split["valid_seen"],
        "valid_unseen": by_split["valid_unseen"],
    }
    if verify_solvable:
        for split in ("valid_seen", "valid_unseen"):
            bad = [r for r in selected[split] if not is_solvable(root, r)]
            if bad:
                raise SystemExit(
                    f"{split}: {len(bad)} games are missing the solvable flag, "
                    f"e.g. {bad[0]['game']}"
                )
    return by_split, selected


def write_lists(root: Path, by_split: dict[str, list[dict]],
                selected: dict[str, list[dict]]) -> None:
    start = legacy_index_start()
    # Official mapping order: ALL train games, then valid_seen, then valid_unseen.
    official: list[dict] = []
    index: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        index[split] = {}
        for rec in by_split[split]:
            if rec["game"] in index[split]:
                raise SystemExit(f"{split}: duplicate game in the scan: {rec['game']}")
            item_id = start + len(official)
            index[split][rec["game"]] = item_id
            official.append({"item_id": item_id, **rec})
    for split, item_path in ITEM_FILES.items():
        items = [{"item_id": f"alfworld_{index[split][r['game']]}"} for r in selected[split]]
        item_path.write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
        print(f"{item_path.relative_to(REPO_ROOT)}: {len(items)} items")
    MAPPINGS_OFFICIAL.write_text(
        json.dumps(official, indent=1) + "\n", encoding="utf-8"
    )
    print(
        f"{MAPPINGS_OFFICIAL.relative_to(REPO_ROOT)}: {len(official)} records "
        f"(index {start}..{start + len(official) - 1})"
    )


def check(root: Path, wanted: list[str], by_split: dict[str, list[dict]],
          selected: dict[str, list[dict]]) -> int:
    """Verify the on-disk frozen files match a fresh computation."""
    problems: list[str] = []
    if not MAPPINGS_OFFICIAL.exists():
        problems.append(f"missing {MAPPINGS_OFFICIAL}")
    else:
        official = json.loads(MAPPINGS_OFFICIAL.read_text(encoding="utf-8"))
        start = legacy_index_start()
        expected = []
        for split in SPLITS:
            for rec in by_split[split]:
                expected.append({"item_id": start + len(expected), **rec})
        if official != expected:
            problems.append(
                f"{MAPPINGS_OFFICIAL.name} differs from a fresh scan "
                f"({len(official)} vs {len(expected)} records)"
            )
    for split, item_path in ITEM_FILES.items():
        if not item_path.exists():
            problems.append(f"missing {item_path}")
            continue
        items = json.loads(item_path.read_text(encoding="utf-8"))
        want = len(selected[split])
        if len(items) != want:
            problems.append(f"{item_path.name}: {len(items)} items, expected {want}")
        ids = []
        for item in items:
            try:
                ids.append(int(str(item["item_id"]).rsplit("_", 1)[1]))
            except (KeyError, ValueError):
                problems.append(f"{item_path.name}: malformed item {item!r}")
        if len(set(ids)) != len(ids):
            problems.append(f"{item_path.name}: duplicate item ids")
    for split in EXPECTED:
        print(f"{split}: {len(by_split[split])} official games, {len(selected[split])} selected")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print("check OK")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default=None,
                        help="ALFWorld data root (default: $ALFWORLD_DATA)")
    parser.add_argument("--base-config", default=str(BASE_CONFIG),
                        help="ALFWorld base config (env.task_types)")
    parser.add_argument("--check", action="store_true",
                        help="Verify the frozen files against a fresh scan; write nothing")
    parser.add_argument("--refresh", action="store_true",
                        help="Rescan the mounted dataset instead of using the /tmp scan cache")
    parser.add_argument("--verify-solvable", action="store_true",
                        help="Read the selected games and require solvable=true (slow on the mount)")
    args = parser.parse_args(argv)

    root = data_root(args.data_root)
    wanted = enabled_types(Path(args.base_config))
    print(f"ALFWorld data root: {root}")
    print(f"task types: {', '.join(wanted)}")
    by_split, selected = build(root, wanted, refresh=args.refresh,
                               verify_solvable=args.verify_solvable)
    if args.check:
        return check(root, wanted, by_split, selected)
    write_lists(root, by_split, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
