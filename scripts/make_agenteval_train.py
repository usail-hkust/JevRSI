#!/usr/bin/env python
"""ONE-TIME generator of the AgentEval train item lists (100 per environment).

The eval protocol keeps the vendored AgentEval *test* lists; the train phase
(the train phase) runs on these 100-item lists. Like the test lists
they are FROZEN inputs once written: episode ids are positional over them,
so never regenerate or reorder. This script exists for provenance only --
it is not part of any run pipeline.

Selection (deterministic, no RNG):
- alfworld: 100 items from mappings_train (data_idx 0..2419), stratified by
  the six task types to mirror the test mix, evenly spread per type.
- minigrid: 100 fresh (level, seed) pairs -- same 40-level table, level
  mix mirroring the test items (the test list covers 18 of the 40 slots),
  smallest seeds not used by the test set.
- textcraft: 100 goals evenly spread over the non-test part of the 544-goal
  depth-ascending space (the test list itself is scattered over the whole
  range, so the train mix mirrors its depth distribution).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jevmaster.envs.agenteval import load_item_indices  # noqa: E402

TRAIN_SIZE = 100


def largest_remainder(counts: dict, total: int) -> dict:
    """Split ``total`` over ``counts`` proportionally (largest remainder)."""
    s = sum(counts.values())
    out = {k: (total * c) // s for k, c in counts.items()}
    rem = total - sum(out.values())
    for k in sorted(counts, key=lambda k: (-(total * counts[k] % s), k))[:rem]:
        out[k] += 1
    return out


def spread(pool: list, n: int) -> list:
    """``n`` evenly spaced values from a sorted ``pool``."""
    if n <= 0:
        return []
    if n == 1:
        return [pool[len(pool) // 2]]
    return [pool[round(i * (len(pool) - 1) / (n - 1))] for i in range(n)]


def interleave(groups: dict) -> list:
    """Round-robin across groups so tasks mix along the run (like the test
    lists): skills for one task accumulate while its other episodes wait."""
    out, queues = [], [list(v) for v in groups.values()]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def write(path: Path, prefix: str, indices: list):
    path.write_text(json.dumps([{"item_id": f"{prefix}_{i}"} for i in indices], indent=1),
                    encoding="utf-8")
    print(f"{path.name}: {len(indices)} items")


def gen_alfworld(out: Path):
    test = set(load_item_indices(str(ROOT / "data/agenteval/alfworld_test.json")))
    train = json.loads((ROOT / "data/agenteval/alfworld_mappings_train.json").read_text())
    root = ROOT / "alfworld_data"
    by_type = defaultdict(list)
    for rec in train:  # item_id is the data_idx; task_type "<six types>-obj-..."
        # The local ALFWORLD_DATA is partial (239 of 2420 train games missing
        # on this pod): sample only over games that exist locally.
        game = (root / "json_2.1.1" / "train" / rec["task_type"]
                / rec["task_id"] / "game.tw-pddl")
        if rec["item_id"] not in test and game.exists():
            by_type[rec["task_type"].split("-")[0]].append(rec["item_id"])
    test_mix = Counter(rec["task_type"].split("-")[0]
                       for rec in json.loads((ROOT / "data/agenteval/alfworld_mappings_test.json").read_text()))
    alloc = largest_remainder({t: test_mix.get(t, 0) for t in by_type}, TRAIN_SIZE)
    picked = {t: spread(by_type[t], n) for t, n in alloc.items()}
    write(out, "alfworld", interleave(picked))


def gen_minigrid(out: Path):
    from jevmaster.envs.minigrid_env import AGENTGYM_LEVELS
    test = load_item_indices(str(ROOT / "data/agenteval/babyai_test.json"))
    slot_of, seed_of = lambda n: n % len(AGENTGYM_LEVELS), lambda n: n // len(AGENTGYM_LEVELS)
    test_slots = Counter(slot_of(n) for n in test)
    used_seeds = defaultdict(set)
    for n in test:
        used_seeds[slot_of(n)].add(seed_of(n))
    alloc = largest_remainder({l: test_slots.get(l, 0) for l in range(len(AGENTGYM_LEVELS))},
                              TRAIN_SIZE)
    picked = {}
    for slot, n in alloc.items():
        seeds = (s for s in range(1024) if s not in used_seeds[slot])
        picked[slot] = [next(seeds) * len(AGENTGYM_LEVELS) + slot for _ in range(n)]
    write(out, "babyai", interleave(picked))


def gen_textcraft(out: Path):
    test = set(load_item_indices(str(ROOT / "data/agenteval/textcraft_test.json")))
    pool = sorted(i for i in range(544) if i not in test)  # depth-ascending
    write(out, "textcraft", spread(pool, TRAIN_SIZE))


if __name__ == "__main__":
    data = ROOT / "data/agenteval"
    gen_alfworld(data / "alfworld_train.json")
    gen_minigrid(data / "babyai_train.json")
    gen_textcraft(data / "textcraft_train.json")
