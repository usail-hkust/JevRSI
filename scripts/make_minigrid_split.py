#!/usr/bin/env python
"""Build the frozen MiniGrid train / ID / OOD split manifests.

Design (the "MixedTrain/MixedTest" split)
-----------------------------------------
BabyAI's mixed levels sample the **task kind** (goto / pickup / open /
putnext / seq) and the map from one reset seed. The fork chooses the task kind
in ``__init__`` with an unseeded RNG, so the adapter pre-seeds that
construction RNG with the episode seed (``make_mixed_env``); a seed is then a
complete, reproducible episode definition.

    split  environment                    count  role
    train  BabyAI-MixedTrainLocal-v0      200    skill accumulation
    id     BabyAI-MixedTrainLocal-v0      150    new seeds -> new maps,
                                                  identical task-kind mix
    ood    BabyAI-MixedTestLocal-v0       150    the held-out color-object
                                                  combinations (yellow box,
                                                  red door/key, green ball,
                                                  grey door) and the unseen
                                                  pickup-then-pickup sequences

This is a custom 200/150/150 subsample, not an AgentGym/AgentEval list. The
per-split task-kind quotas are exact and identical across train / ID / OOD
(40/40/40/40/40, 30 each, 30 each), so ID and OOD isolate the map / object
shift rather than a different action-kind mix. Seeds come from disjoint
blocks (train 0+, ID 1M+, OOD 2M+), and a map+agent fingerprint is deduped
across ALL splits, so ID cannot reuse a train map.

The held-out combinations are the ones the MixedTrain level excludes from its
instructions (``mixed_seq_levels.Level_MixedTrainLocal`` / ``Level_MixedTestLocal``):
yellow box, red door/key, green ball, grey door. The generator asserts that
train/ID missions never mention them and every OOD mission does.

Determinism
-----------
Every item is played once at generation time and recorded (mission, task kind,
grid size, objects, map+agent fingerprint). ``--check`` validates the frozen
files structurally without MiniGrid; ``--check --verify-loadable`` replays all
items and compares them; ``--check --verify-stability`` replays each split in
fresh processes (different PYTHONHASHSEED) and requires identical
observations/candidate lists.

Usage
-----
    # Delivered pilot split (200 / 150 / 150; the default)
    /opt/python/bin/python scripts/make_minigrid_split.py

    # Validate without MiniGrid
    /opt/python/bin/python scripts/make_minigrid_split.py --check

    # Replay every frozen item (MiniGrid + gymnasium)
    ... --check --verify-loadable

    # Cross-process replay (fresh processes, different PYTHONHASHSEED)
    ... --check --verify-stability

Outputs (tracked):
    data/minigrid/splits/minigrid_manifest.json
    data/minigrid/splits/minigrid_train.json
    data/minigrid/splits/minigrid_id.json
    data/minigrid/splits/minigrid_ood.json
    data/minigrid/splits/README.md
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data/minigrid/splits"
SCHEMA = "jevmaster.split/v1"
SPLIT_ORDER: Tuple[str, ...] = ("train", "id", "ood")

ENV_TRAIN = "BabyAI-MixedTrainLocal-v0"
ENV_TEST = "BabyAI-MixedTestLocal-v0"
ENV_FOR_SPLIT = {"train": ENV_TRAIN, "id": ENV_TRAIN, "ood": ENV_TEST}
ENV_SHORT = {"train": "mixedtrain", "id": "mixedtrain", "ood": "mixedtest"}
KINDS: Tuple[str, ...] = ("goto", "pickup", "open", "putnext", "pick up seq go to")

# The instruction combinations MixedTrainLocal deliberately withholds and
# MixedTestLocal is built around.
HELD_OUT_COMBOS: Tuple[Tuple[str, str], ...] = (
    ("yellow", "box"),
    ("red", "door"),
    ("red", "key"),
    ("green", "ball"),
    ("grey", "door"),
)

PROFILES: Dict[str, Dict[str, int]] = {
    "pilot": {"train": 200, "id": 150, "ood": 150},
}
SEED_BASE: Dict[str, int] = {"train": 0, "id": 1_000_000, "ood": 2_000_000}
SEED_BLOCK = 1_000_000


# ---------------------------------------------------------------------------
# Pure allocation (no MiniGrid needed)
# ---------------------------------------------------------------------------
def distribute(total: int, kinds: Sequence[str] = KINDS) -> Dict[str, int]:
    """Even per-kind counts (remainder to the first kinds)."""
    base, rem = divmod(int(total), len(kinds))
    return {kind: base + (1 if i < rem else 0) for i, kind in enumerate(kinds)}


def quotas_for(counts: Mapping[str, int]) -> Dict[str, Dict[str, int]]:
    return {split: distribute(counts[split]) for split in SPLIT_ORDER }


def seed_range(split: str) -> Tuple[int, int]:
    start = SEED_BASE[split]
    return start, start + SEED_BLOCK - 1


def mission_has_held_out(mission: str) -> bool:
    text = " " + str(mission).lower() + " "
    return any(f" {color} {obj} " in text for color, obj in HELD_OUT_COMBOS)


def item_id(split: str, seed: int) -> str:
    return f"minigrid_{ENV_SHORT[split]}_{split}_s{int(seed)}"


def fingerprint(grid: Sequence, agent: Sequence[int]) -> str:
    payload = json.dumps({"grid": grid, "agent": [int(v) for v in agent]},
                         sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Episode construction (MiniGrid needed)
# ---------------------------------------------------------------------------
def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def capture(env_id: str, seed: int) -> Dict[str, Any]:
    """Play one fresh, seed-constructed episode and record its definition."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from jevmaster.envs.minigrid_env import make_mixed_env

    with _quiet():
        env = make_mixed_env(env_id, seed)
        try:
            observation, info = env.reset(seed=int(seed))
            raw = env.unwrapped
            mission = str(observation.get("mission") or "")
            kinds = getattr(raw, "action_kinds", None) or ()
            kind = str(kinds[0]) if kinds else ""
            if kind not in KINDS:
                raise ValueError(f"{env_id}/{seed}: unexpected action kind {kind!r}")
            if not mission.strip():
                raise ValueError(f"{env_id}/{seed}: empty mission")
            width, height = int(raw.width), int(raw.height)
            agent = [int(raw.agent_pos[0]), int(raw.agent_pos[1]),
                     int(raw.agent_dir)]
            grid = raw.grid.encode().tolist()
            objects = set()
            for cell in raw.grid.grid:
                if cell is not None:
                    objects.add((str(cell.color), str(cell.type)))
        finally:
            env.close()
    return {
        "env_id": env_id,
        "seed": int(seed),
        "action_kind": kind,
        "mission": mission,
        "grid_size": [width, height],
        "agent": agent,
        "objects": [list(pair) for pair in sorted(objects)],
        "fingerprint": fingerprint(grid, agent),
    }


def select_split(split: str, quotas: Mapping[str, int],
                 seen_fingerprints: set) -> Tuple[List[Dict[str, Any]],
                                                  Dict[str, Any]]:
    """Scan the split's seed block for exact kind quotas and unseen maps."""
    env_id = ENV_FOR_SPLIT[split]
    start, end = seed_range(split)
    needed = {kind: int(quotas[kind]) for kind in KINDS}
    picked: Dict[str, List[Dict[str, Any]]] = {kind: [] for kind in KINDS}
    stats = {"attempts": 0, "failures": 0, "duplicate_maps": 0,
             "seed_start": start, "seed_end": None, "per_kind": {}}
    target = sum(int(quotas[kind]) for kind in KINDS)
    seed = start
    while sum(len(rows) for rows in picked.values()) < target:
        if seed > end:
            raise RuntimeError(f"{split}: exhausted seed block [{start}, {end}]")
        stats["attempts"] += 1
        try:
            candidate = capture(env_id, seed)
        except Exception as exc:
            stats["failures"] += 1
            if stats["failures"] <= 5 or stats["failures"] % 50 == 0:
                print(f"[{split}] seed {seed} rejected: "
                      f"{type(exc).__name__}: {exc}", flush=True)
            seed += 1
            continue
        kind = candidate["action_kind"]
        if candidate["fingerprint"] in seen_fingerprints:
            stats["duplicate_maps"] += 1
        elif needed.get(kind, 0) > 0:
            candidate["split"] = split
            candidate["item_id"] = item_id(split, seed)
            candidate["source"] = env_id
            picked[kind].append(candidate)
            seen_fingerprints.add(candidate["fingerprint"])
            needed[kind] -= 1
            if sum(len(rows) for rows in picked.values()) % 25 == 0:
                print(f"[{split}] {sum(len(rows) for rows in picked.values())}"
                      f"/{sum(quotas.values())} (seed {seed})", flush=True)
        seed += 1
    stats["seed_end"] = int(seed - 1)
    stats["per_kind"] = {kind: len(rows) for kind, rows in picked.items()}
    rows = [row for kind in KINDS for row in picked[kind]]
    return rows, stats


def ordered_items(items: Sequence[dict]) -> List[dict]:
    """Split order, then round-robin across task kinds within each split."""
    out: List[dict] = []
    for split in SPLIT_ORDER:
        rows = [item for item in items if item["split"] == split]
        by_kind = {kind: sorted((item for item in rows
                                 if item["action_kind"] == kind),
                                key=lambda item: int(item["seed"]))
                   for kind in KINDS}
        max_len = max((len(by_kind[kind]) for kind in KINDS), default=0)
        for index in range(max_len):
            for kind in KINDS:
                if index < len(by_kind[kind]):
                    out.append(by_kind[kind][index])
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def make_item(candidate: Mapping[str, Any], split: str) -> Dict[str, Any]:
    return {
        "item_id": item_id(split, int(candidate["seed"])),
        "split": split,
        "task": ENV_FOR_SPLIT[split],
        "variation": int(candidate["seed"]),
        "seed": int(candidate["seed"]),
        "action_kind": str(candidate["action_kind"]),
        "mission": str(candidate["mission"]),
        "grid_size": [int(v) for v in candidate["grid_size"]],
        "objects": [[str(c), str(t)] for c, t in candidate["objects"]],
        "fingerprint": str(candidate["fingerprint"]),
        "source": ENV_FOR_SPLIT[split],
    }


def validate_manifest(manifest: Mapping[str, Any], check_counts: bool = True) -> Dict[str, Any]:
    problems: List[str] = []
    if manifest.get("schema") != SCHEMA:
        problems.append(f"schema {manifest.get('schema')!r} != {SCHEMA!r}")
    if manifest.get("env") != "minigrid":
        problems.append(f"env {manifest.get('env')!r} != 'minigrid'")
    counts = manifest.get("counts") or {}
    if set(counts) != set(SPLIT_ORDER):
        problems.append(f"counts keys {sorted(counts)} != {list(SPLIT_ORDER)}")
    quotas = manifest.get("quotas_per_kind") or {}
    items = manifest.get("items") or []
    for split in SPLIT_ORDER:
        row = quotas.get(split) or {}
        if set(row) != set(KINDS):
            problems.append(f"{split}: quota kinds {sorted(row)} != {list(KINDS)}")
        elif sum(int(v) for v in row.values()) != int(counts.get(split, -1)):
            problems.append(f"{split}: quotas do not sum to the split count")

    seen_ids: set = set()
    seen_seeds: set = set()
    seen_fingerprints: set = set()
    per_split: Dict[str, List[dict]] = {split: [] for split in SPLIT_ORDER}
    for index, item in enumerate(items):
        split = str(item.get("split") or "")
        if split not in SPLIT_ORDER:
            problems.append(f"item {index}: bad split {split!r}")
            continue
        if str(item.get("task")) != ENV_FOR_SPLIT[split]:
            problems.append(
                f"item {index}: task {item.get('task')!r} != "
                f"{ENV_FOR_SPLIT[split]!r}")
        if item.get("source") not in (None, ENV_FOR_SPLIT[split]):
            problems.append(f"item {index}: source {item.get('source')!r} mismatch")
        seed = item.get("seed", item.get("variation"))
        if not isinstance(seed, int):
            problems.append(f"item {index}: seed {seed!r} is not an int")
            continue
        start, end = seed_range(split)
        if not start <= seed <= end:
            problems.append(f"item {index}: seed {seed} outside [{start}, {end}]")
        if item.get("variation") != seed:
            problems.append(f"item {index}: variation != seed")
        kind = str(item.get("action_kind") or "")
        if kind not in KINDS:
            problems.append(f"item {index}: bad action_kind {kind!r}")
        mission = str(item.get("mission") or "")
        if not mission.strip():
            problems.append(f"item {index}: empty mission")
        held = mission_has_held_out(mission)
        if split == "ood" and not held:
            problems.append(f"item {index}: OOD mission misses held-out combos: "
                            f"{mission!r}")
        if split in ("train", "id") and held:
            problems.append(f"item {index}: train/ID mission leaks held-out "
                            f"combos: {mission!r}")
        if item.get("item_id") in seen_ids:
            problems.append(f"item {index}: duplicate item_id")
        seen_ids.add(item.get("item_id"))
        if (ENV_FOR_SPLIT[split], seed) in seen_seeds:
            problems.append(f"item {index}: duplicate (task, seed)")
        seen_seeds.add((ENV_FOR_SPLIT[split], seed))
        if item.get("fingerprint") in seen_fingerprints:
            problems.append(f"item {index}: duplicate map fingerprint")
        seen_fingerprints.add(item.get("fingerprint"))
        per_split[split].append(item)

    for split in SPLIT_ORDER:
        rows = per_split[split]
        if check_counts and len(rows) != int(counts.get(split, -1)):
            problems.append(f"{split}: {len(rows)} items, expected {counts.get(split)}")
        want = {kind: int((quotas.get(split) or {}).get(kind, -1))
                for kind in KINDS}
        got = {kind: sum(1 for row in rows if row["action_kind"] == kind)
               for kind in KINDS}
        if check_counts and got != want:
            problems.append(f"{split}: kind counts {got} != {want}")
        # ID and OOD keep the train task-kind *shares* (not the raw counts).
        if split in ("id", "ood") and quotas.get("train"):
            for kind in KINDS:
                share = got.get(kind, 0) / max(1, int(counts.get(split, 1)))
                train_share = (int((quotas.get("train") or {}).get(kind, 0))
                               / max(1, int(counts.get("train", 1))))
                if abs(share - train_share) > 0.01:
                    problems.append(
                        f"{split}: {kind} share {share:.3f} differs from train "
                        f"{train_share:.3f}")
        # Round-robin: the first block is one item per non-empty kind.
        expected_first = [kind for kind in KINDS
                          if int((quotas.get(split) or {}).get(kind, 0)) > 0]
        first = [row["action_kind"] for row in rows[:len(expected_first)]]
        if expected_first and first != expected_first:
            problems.append(f"{split}: items are not kind-interleaved: {first}")

    train_seeds = {int(row["seed"]) for row in per_split["train"]}
    id_seeds = {int(row["seed"]) for row in per_split["id"]}
    if train_seeds & id_seeds:
        problems.append("train and ID share seeds")

    if problems:
        raise ValueError("MiniGrid split validation failed:\n  - "
                         + "\n  - ".join(problems[:25]))
    return {"counts": {split: len(per_split[split]) for split in SPLIT_ORDER}}


def _readme(manifest: Mapping[str, Any]) -> str:
    counts = manifest["counts"]
    quotas = manifest["quotas_per_kind"]
    generation = manifest.get("generation") or {}
    lines = [
        "# MiniGrid / BabyAI split manifests (MixedTrain/MixedTest)",
        "",
        "Generated by `scripts/make_minigrid_split.py`; do not edit by hand.",
        "",
        "| split | environment | count | task kinds | role |",
        "|---|---|---|---|---|",
        f"| train | `{ENV_TRAIN}` | {counts['train']} | "
        f"{'/'.join(str(quotas['train'][k]) for k in KINDS)} | skill accumulation |",
        f"| id | `{ENV_TRAIN}` | {counts['id']} | "
        f"{'/'.join(str(quotas['id'][k]) for k in KINDS)} | new seeds -> new maps, "
        "same task distribution |",
        f"| ood | `{ENV_TEST}` | {counts['ood']} | "
        f"{'/'.join(str(quotas['ood'][k]) for k in KINDS)} | held-out color-object "
        "combinations |",
        "",
        "Task kinds order: " + ", ".join(f"`{kind}`" for kind in KINDS) + ".",
        "",
        "The MixedTrain env chooses its task kind in `__init__`; the adapter",
        "pre-seeds that construction RNG with the episode seed, so a",
        "`(env, seed)` pair is one reproducible episode. Seeds come from",
        "disjoint blocks (train 0+, ID 1M+, OOD 2M+), and map+agent",
        "fingerprints are deduped across all splits, so ID never reuses a",
        "train map.",
        "",
        "OOD uses the combinations MixedTrainLocal withholds from its",
        "instructions: yellow box, red door, red key, green ball, grey door",
        "(and the test-only pickup-then-pickup sequences). `--check` asserts",
        "that no train/ID mission mentions them and every OOD mission does.",
        "",
        "| split | seed block | attempts | failures | duplicate maps |",
        "|---|---|---|---|---|",
    ]
    stats = generation.get("stats") or {}
    for split in SPLIT_ORDER:
        row = stats.get(split) or {}
        start, end = seed_range(split)
        lines.append(
            f"| {split} | {start:,} - {end:,} | {row.get('attempts', '?')} | "
            f"{row.get('failures', '?')} | {row.get('duplicate_maps', '?')} |")
    lines += [
        "",
        "Each item stores the mission, task kind, grid size, objects and a",
        "map+agent fingerprint. Files are round-robin interleaved across task",
        "kinds, so a prefix covers every kind.",
        "",
        "`scripts/run_protocol.py --config configs/minigrid.yaml` runs",
        "train -> eval-id -> eval-ood; both eval phases start from the same",
        "post-train skill-store snapshot. `--check` works without MiniGrid;",
        "`--check --verify-loadable` replays every item;",
        "`--check --verify-stability` replays each split in fresh processes.",
        "",
        f"MiniGrid version used: {manifest.get('minigrid_version')}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build / check
# ---------------------------------------------------------------------------
def _env_version() -> str:
    try:
        import importlib.metadata
        return str(importlib.metadata.version("minigrid"))
    except Exception:
        try:
            import minigrid
            return str(getattr(minigrid, "__version__", "unknown"))
        except Exception:
            return "unknown"


def build(counts: Mapping[str, int]) -> Tuple[dict, Dict[str, List[dict]]]:
    try:
        import gymnasium  # noqa: F401
        import minigrid
        minigrid.register_minigrid_envs()
    except ImportError as exc:
        raise SystemExit(
            "minigrid/gymnasium are not installed for this interpreter; install "
            "the BALROG BabyAI-Text fork (git clone "
            "https://github.com/BartekCupial/Minigrid && pip install -e "
            "./Minigrid) and run with /opt/python/bin/python"
        ) from exc
    quotas = quotas_for(counts)
    seen_fingerprints: set = set()
    items: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    for split in SPLIT_ORDER:
        print(f"[{split}] scanning {ENV_FOR_SPLIT[split]} from "
              f"seed {SEED_BASE[split]:,} for "
              f"{'/'.join(str(quotas[split][k]) for k in KINDS)}", flush=True)
        rows, split_stats = select_split(split, quotas[split], seen_fingerprints)
        items.extend(make_item(row, split) for row in rows)
        stats[split] = split_stats
        print(f"[{split}] done: {len(rows)} items, seeds "
              f"{split_stats['seed_start']}..{split_stats['seed_end']}", flush=True)
    items = ordered_items(items)
    manifest = {
        "schema": SCHEMA,
        "env": "minigrid",
        "minigrid_version": _env_version(),
        "profile": "pilot" if counts == PROFILES["pilot"] else "custom",
        "counts": dict(counts),
        "quotas_per_kind": quotas,
        "task_kinds": list(KINDS),
        "seed_blocks": {split: list(seed_range(split)) for split in SPLIT_ORDER},
        "env_per_split": dict(ENV_FOR_SPLIT),
        "held_out_combos": [list(pair) for pair in HELD_OUT_COMBOS],
        "selection_rule": (
            "scan the split's seed block; accept a seed when its seeded "
            "construction action kind still needs quota and its map+agent "
            "fingerprint is unseen across all splits"
        ),
        "generation": {"stats": stats, "load_failures": []},
        "items": items,
    }
    validate_manifest(manifest)
    return manifest, {
        split: [item for item in items if item["split"] == split]
        for split in SPLIT_ORDER
    }


def load_frozen() -> Optional[dict]:
    path = OUT_DIR / "minigrid_manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def write_outputs(manifest: Mapping[str, Any],
                  per_split: Mapping[str, List[dict]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "minigrid_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for split in SPLIT_ORDER:
        path = OUT_DIR / f"minigrid_{split}.json"
        path.write_text(json.dumps(per_split[split], indent=1, ensure_ascii=False)
                        + "\n", encoding="utf-8")
        print(f"{path.relative_to(REPO_ROOT)}: {len(per_split[split])} items")
    (OUT_DIR / "README.md").write_text(_readme(manifest), encoding="utf-8")
    print(f"{(OUT_DIR / 'README.md').relative_to(REPO_ROOT)}")


def check_outputs(verify_loadable: bool = False,
                  verify_stability: bool = False,
                  stability_repeats: int = 5) -> int:
    manifest = load_frozen()
    if manifest is None:
        print(f"missing {OUT_DIR / 'minigrid_manifest.json'}", file=sys.stderr)
        return 1
    try:
        stats = validate_manifest(manifest)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for split in SPLIT_ORDER:
        path = OUT_DIR / f"minigrid_{split}.json"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1
        rows = json.loads(path.read_text(encoding="utf-8"))
        expected = [item for item in manifest["items"] if item["split"] == split]
        if rows != expected:
            print(f"{path.name} differs from the manifest", file=sys.stderr)
            return 1
    if verify_loadable and not _verify_loadable(manifest):
        return 1
    if verify_stability and not _verify_stability(manifest, stability_repeats):
        return 1
    print("MiniGrid split check OK")
    for split in SPLIT_ORDER:
        print(f"  {split}: {stats['counts'][split]} items")
    return 0


def _verify_loadable(manifest: Mapping[str, Any]) -> bool:
    try:
        import gymnasium  # noqa: F401
        import minigrid  # noqa: F401
    except ImportError:
        print("minigrid/gymnasium are not installed; cannot --verify-loadable",
              file=sys.stderr)
        return False
    bad = 0
    for item in manifest.get("items") or []:
        try:
            candidate = capture(str(item["task"]), int(item["seed"]))
        except Exception as exc:
            print(f"{item['item_id']}: load failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            bad += 1
            continue
        for field in ("action_kind", "mission", "grid_size", "objects",
                      "fingerprint"):
            if candidate[field] != item.get(field):
                print(f"{item['item_id']}: {field} differs from the manifest",
                      file=sys.stderr)
                bad += 1
    if bad:
        print(f"{bad} item(s) failed load verification", file=sys.stderr)
        return False
    print(f"replayed {len(manifest.get('items') or [])} items; definitions match")
    return True


# ---------------------------------------------------------------------------
# Cross-process stability probe
# ---------------------------------------------------------------------------
PROBE_MARKER = "PROBE_JSON:"


def probe_worker(items_path: str) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from jevmaster.envs.minigrid_env import MiniGridEnvAdapter

    records = json.loads(Path(items_path).read_text(encoding="utf-8"))
    env = MiniGridEnvAdapter(items_file=str(items_path))
    out: List[Dict[str, Any]] = []
    try:
        for record in records:
            entry = {
                "task": str(record["task"]),
                "variation": int(record["variation"]),
                "item_id": str(record.get("item_id") or ""),
            }
            try:
                with _quiet():
                    observation = env.reset(entry["task"], entry["variation"])
                    candidates = env.candidates()
                entry.update({
                    "goal": str(observation.goal or ""),
                    "text": str(observation.text or ""),
                    "inventory": str(observation.inventory or ""),
                    "action_kind": str(observation.extra.get("action_kind") or ""),
                    "candidates": [[c.action, c.template] for c in candidates],
                })
                if candidates:
                    with _quiet():
                        step = env.step(candidates[0].action)
                    entry["step_text"] = str(step.observation.text or "")
                    entry["step_inventory"] = str(step.observation.inventory or "")
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            out.append(entry)
    finally:
        env.close()
    print(PROBE_MARKER + json.dumps({"items": out}, sort_keys=True))
    return 0


def _signature(entry: Mapping[str, Any]):
    return (
        str(entry.get("goal") or ""),
        str(entry.get("text") or ""),
        str(entry.get("inventory") or ""),
        str(entry.get("action_kind") or ""),
        tuple(tuple(str(part) for part in pair)
              for pair in entry.get("candidates") or []),
        str(entry.get("step_text") or ""),
        str(entry.get("step_inventory") or ""),
        str(entry.get("error") or ""),
    )


def run_probe(records: Sequence[Mapping[str, Any]], repeats: int = 5,
              workers: int = 5, timeout: float = 1800.0
              ) -> List[List[Dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="mg-stability-") as tmp:
        path = Path(tmp) / "items.json"
        path.write_text(json.dumps(list(records)), encoding="utf-8")

        def one(seed: int):
            env = dict(os.environ, PYTHONHASHSEED=str(seed))
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--probe-items", str(path)],
                cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                timeout=timeout,
            )
            lines = [line for line in proc.stdout.splitlines()
                     if line.startswith(PROBE_MARKER)]
            if proc.returncode != 0 or not lines:
                raise RuntimeError(
                    f"stability probe failed (seed {seed}, exit "
                    f"{proc.returncode}): {proc.stderr[-1500:]}")
            return json.loads(lines[-1][len(PROBE_MARKER):])["items"]

        with ThreadPoolExecutor(max_workers=max(1, min(workers, repeats))) as pool:
            futures = [pool.submit(one, seed) for seed in range(1, repeats + 1)]
            results = [future.result() for future in futures]
    lengths = {len(result) for result in results}
    if len(lengths) != 1 or (lengths and next(iter(lengths)) != len(records)):
        raise RuntimeError(f"stability probe returned inconsistent item counts: "
                           f"{sorted(lengths)} vs {len(records)}")
    return results


def _verify_stability(manifest: Mapping[str, Any], repeats: int = 5) -> bool:
    try:
        import gymnasium  # noqa: F401
        import minigrid  # noqa: F401
    except ImportError:
        print("minigrid/gymnasium are not installed; cannot --verify-stability",
              file=sys.stderr)
        return False
    bad = 0
    items = manifest.get("items") or []
    for split in SPLIT_ORDER:
        rows = [item for item in items if item["split"] == split]
        if not rows:
            continue
        results = run_probe(rows, repeats=repeats, workers=5)
        for index, item in enumerate(rows):
            signatures = [_signature(result[index]) for result in results]
            if len(set(signatures)) > 1 or signatures[0][-1]:
                print(f"{item['item_id']}: unstable across fresh processes",
                      file=sys.stderr)
                bad += 1
            elif signatures[0][0] != item["mission"]:
                print(f"{item['item_id']}: probe mission differs from the "
                      f"manifest", file=sys.stderr)
                bad += 1
    if bad:
        print(f"{bad} item(s) failed stability verification", file=sys.stderr)
        return False
    print(f"replayed {len(items)} items in {repeats} fresh processes; stable")
    return True


def resolve_counts(args) -> Tuple[Dict[str, int], str]:
    base = dict(PROFILES[args.profile])
    overridden = False
    for name, value in (("train", args.train), ("id", args.id), ("ood", args.ood)):
        if value is not None:
            base[name] = int(value)
            overridden = True
    return base, "custom" if overridden else args.profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="pilot",
                        help="count profile (default: pilot = 200/150/150)")
    parser.add_argument("--train", type=int, default=None)
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--ood", type=int, default=None)
    parser.add_argument("--check", action="store_true",
                        help="validate the frozen files and exit")
    parser.add_argument("--verify-loadable", action="store_true",
                        help="with --check: replay every frozen item")
    parser.add_argument("--verify-stability", action="store_true",
                        help="with --check: replay each split in fresh processes")
    parser.add_argument("--stability-repeats", type=int, default=5,
                        help="fresh processes per split (default 5)")
    parser.add_argument("--probe-items", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe_items:
        return probe_worker(args.probe_items)

    counts, profile = resolve_counts(args)
    if any(v < 0 for v in counts.values()):
        parser.error("counts must be >= 0")
    if args.check:
        return check_outputs(verify_loadable=args.verify_loadable,
                             verify_stability=args.verify_stability,
                             stability_repeats=args.stability_repeats)

    manifest, per_split = build(counts)
    manifest["profile"] = profile
    write_outputs(manifest, per_split)
    stats = manifest["generation"]["stats"]
    print(f"\nDone: {len(manifest['items'])} items "
          f"(train {counts['train']}, id {counts['id']}, ood {counts['ood']})")
    for split in SPLIT_ORDER:
        row = stats[split]
        print(f"  {split}: seeds {row['seed_start']}..{row['seed_end']}, "
              f"attempts {row['attempts']}, failures {row['failures']}, "
              f"duplicate maps {row['duplicate_maps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
