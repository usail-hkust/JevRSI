"""Frozen split manifests for the train -> eval-id -> eval-ood protocol.

A *split manifest* is a JSON array of item records, one record per episode.
Every record carries a stable ``item_id``; the rest is environment-specific.
The record schema intentionally stays open so each adapter can add the fields
its episodes need (ALFWorld: the AgentEval ``data_idx`` derived from
``item_id``;
``task`` / ``level`` / ``seed`` / ``goal`` / ...).

The protocol uses three manifests per benchmark::

    train   the warm-up; System-2 may take over and the skill store grows
    id      in-distribution eval; skills start from a frozen post-train copy
    ood     out-of-distribution eval; same frozen post-train starting point

``split_phase_plan`` is the single place that turns those three files into the
phase tuples consumed by ``scripts/run_protocol.py``. Every environment that
adopts the ID/OOD protocol uses the same plan, so the record directories,
tracker prefixes and skill snapshots are identical across benchmarks.

The loader validates only what the protocol relies on: records are objects,
``item_id`` is present and unique, and (when a split is requested) that split
has at least one record (the file is filtered to it). An optional ``seed``
field is also checked for duplicates within a ``(task, seed)`` key because two
episodes generating the same game would silently duplicate evaluation data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

SPLITS: Tuple[str, ...] = ("train", "id", "ood")

# Record-dir tag and tracker prefix per split. Kept identical for every
# benchmark so runs and dashboards stay comparable.
PHASE_TAGS: Dict[str, str] = {"train": "train", "id": "eval-id", "ood": "eval-ood"}
PHASE_PREFIXES: Dict[str, str] = {
    "train": "train",
    "id": "eval/id",
    "ood": "eval/ood",
}


class SplitManifestError(ValueError):
    """Malformed or inconsistent split manifest."""


def _as_items(data: Any, path: Path) -> List[Dict[str, Any]]:
    """Accept a raw item array or a ``{"items": [...]}`` wrapper."""
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list):
        raise SplitManifestError(
            f"{path}: expected a JSON array of item records or an object with "
            f"an 'items' array"
        )
    return data


def load_items(path: str | Path, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load, validate and (optionally) filter one split manifest.

    The whole file is validated first (unique ``item_id`` / ``seed``), then
    ``split`` selects the records whose ``split`` field matches; records
    without a ``split`` field (the AgentEval lists) are kept. An empty
    selection is an error, so a typo cannot silently run zero episodes.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"split manifest not found: {p} (run from the repo root or pass an "
            f"absolute path)"
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:  # includes JSONDecodeError
        raise SplitManifestError(f"{p}: invalid JSON: {exc}") from exc

    items = _as_items(data, p)
    seen_ids: set[str] = set()
    seen_seeds: set[tuple[str, Any]] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise SplitManifestError(f"{p}:{i}: item is not an object: {item!r}")
        item_id = str(item.get("item_id") or "").strip()
        if not item_id:
            raise SplitManifestError(f"{p}:{i}: missing item_id")
        if item_id in seen_ids:
            raise SplitManifestError(f"{p}:{i}: duplicate item_id {item_id!r}")
        seen_ids.add(item_id)
        if "seed" in item:
            key = (str(item.get("task") or ""), item["seed"])
            if key in seen_seeds:
                raise SplitManifestError(
                    f"{p}:{i}: duplicate seed {item['seed']!r} for task "
                    f"{key[0]!r} (two episodes would generate the same game)"
                )
            seen_seeds.add(key)
    if split is not None:
        items = [item for item in items
                 if item.get("split") is None or str(item["split"]) == split]
        if not items:
            raise SplitManifestError(f"{p}: no items for split {split!r}")
    return items


def split_phase_plan(paths: Mapping[str, str]) -> List[Tuple[str, str, str, Dict[str, str]]]:
    """Standard phase plan for a benchmark with train / ID / OOD manifests.

    Returns the explicit 4-tuple form understood by
    :func:`scripts.run_protocol.phase_specs`::

        (tag, run.phase, tracker prefix, env.params overrides)

    ``run.phase`` stays the two semantic values the agent knows (train|eval):
    the eval-id / eval-ood tags only name the record directory and metric
    namespace, so plan/pointer binding rules are identical
    for both eval splits.
    """
    missing = [split for split in SPLITS if not paths.get(split)]
    if missing:
        raise SplitManifestError(
            f"split plan is missing manifests for: {', '.join(missing)}"
        )
    return [
        (
            PHASE_TAGS[split],
            "train" if split == "train" else "eval",
            PHASE_PREFIXES[split],
            {"items_file": str(paths[split])},
        )
        for split in SPLITS
    ]


def manifest_counts(items: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    """Count records per split (``unknown`` for records without one)."""
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item.get("split") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts
