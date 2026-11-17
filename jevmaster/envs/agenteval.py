"""AgentGym AgentEval item lists (test + train).

The aligned environments (alfworld, minigrid/BabyAI,
take their EVAL episode set from the AgentGym AgentEval test lists
(huggingface.co/datasets/AgentGym/AgentEval), vendored byte-frozen under
``data/agenteval/``. The TRAIN phase of the two-phase protocol
runs on ``*_train.json``: 100 items per environment, disjoint from the test
list, generated once by ``scripts/make_agenteval_train.py`` and equally
frozen. Each file is a JSON array of ``{"item_id": "<env>_<n>"}`` records;
``n`` is the environment server's ``data_idx`` (AgentGym's
``int(item_id.rsplit("_", 1)[1])``). Episode ids are positional over the
item order, so these files must never be reordered or regenerated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# env name -> vendored item list (paths are repo-root/CWD relative, like the
# other env data_dir params)
ITEM_FILES: Dict[str, str] = {
    "alfworld": "data/agenteval/alfworld_test.json",
    "minigrid": "data/agenteval/babyai_test.json",
}

# The train phase's lists (same schema, disjoint from the test lists).
TRAIN_FILES: Dict[str, str] = {
    "alfworld": "data/agenteval/alfworld_train.json",
    "minigrid": "data/agenteval/babyai_train.json",
}


def item_file(env: str, phase: str) -> str:
    """The vendored item list for ``env``'s ``phase`` ("test" | "train")."""
    if phase == "test":
        return ITEM_FILES[env]
    if phase == "train":
        return TRAIN_FILES[env]
    raise ValueError(f"unknown item-list phase {phase!r} (test | train)")


def load_item_indices(path: str) -> List[int]:
    """Parse an AgentEval item list into data_idx values, in file order.

    ``[{"item_id": "sciworld_3491"}, ...]`` -> ``[3491, ...]``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"AgentEval item list not found: {p} (vendored under data/agenteval/; "
            "run from the repo root or pass env.params.items_file)"
        )
    items = json.loads(p.read_text(encoding="utf-8"))
    indices: List[int] = []
    for i, record in enumerate(items):
        item_id = record.get("item_id") if isinstance(record, dict) else None
        prefix, _, num = str(item_id).rpartition("_")
        try:
            indices.append(int(num))
        except ValueError:
            raise ValueError(
                f"{p}:{i}: malformed AgentEval item_id {item_id!r} "
                "(expected '<env>_<data_idx>')"
            ) from None
    return indices
