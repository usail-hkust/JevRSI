#!/usr/bin/env python
"""Run the AgentEval two-phase protocol in ONE command.

Default split profile: ``agenteval`` -- the M3-era protocol. ALFWorld,
MiniGrid and TextCraft-Synth run train -> eval on the vendored AgentEval
lists in file order (``data/agenteval/<env>_train.json`` -> ``<env>_test.json``:
100 train items, 200/200/90/100 eval items), one skill store across both
phases, and the eval phase keeps accumulating (no snapshot, no freeze).
``--split-profile official`` selects the frozen ID/OOD split manifests instead
(train -> eval-id + eval-ood).

- **train** (``data/agenteval/*_train.json`` under the default profile; the
  official train manifests under ``official``): the warm-up. System-2 reviews
  low-confidence steps and builds the three-component skill library.
- **eval** (the vendored AgentEval test set under the default profile; under
  ``official``: ALFWorld ``valid_seen``/``valid_unseen``, MiniGrid new
  seeds / MixedTestLocal,
  TextCraft-Synth ID/OOD): the reported numbers.
  Under ``official`` System-2 keeps assisting in every phase: both ID and OOD
  start from the SAME post-train skill-store snapshot and continue developing
  from it (``skills.freeze: true`` makes the eval copies read-only instead).

Each phase gets its own records dir (``<base>-train``, ``<base>-eval``;
official ID/OOD benchmarks ``<base>-eval-id`` / ``<base>-eval-ood``) and its
own tracker run with its step axis from 0: per-episode metrics under the phase
prefix (``train/...``, ``eval/...``, ``eval/id/...``, ``eval/ood/...``),
running ``dataset/*`` aggregates and the top-level ``final/<prefix>/*``
report averaged over that phase's episodes only. The command-line final
report prints every phase.

    Default (agenteval, M3 era):
    ALFWorld:      train 100 / eval 200    AgentEval items
    MiniGrid:      train 100 / eval 90     AgentEval items
    TextCraft-Synth: train 200 / eval 150 ID + 150 OOD
    Official (--split-profile official):
    ALFWorld:      train 200 official train / eval 140 ID + 134 OOD
    MiniGrid:      train 200 MixedTrainLocal / eval 150 ID + 150 OOD

Adding a benchmark with ID/OOD splits: produce three manifests (train/id/ood)
and register them with ``jevmaster.envs.splits.split_phase_plan``; the phase
loop, record directories, metric prefixes, post-train snapshot and the
eval-library policy are shared by every dataset.

Examples:
    python scripts/run_protocol.py --config configs/alfworld.yaml --episodes 10
    python scripts/run_protocol.py --config configs/minigrid.yaml

All modes run the same phases with the same episodes (aligned protocol):
jev has System-2 disabled, llm is the pure LLM baseline (LLM System-1,
no System-2/skills/memory), and llm_agent uses the LLM as System-1 with the
System-2 skill components; none changes the episode set.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))        # analyze_protocol
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # jevmaster

import analyze_protocol  # noqa: E402
from jevmaster.agent import build_agent  # noqa: E402
from jevmaster.config import load_config  # noqa: E402
from jevmaster.envs.agenteval import ITEM_FILES, item_file  # noqa: E402
from jevmaster.envs.splits import split_phase_plan  # noqa: E402
from jevmaster.skills.store import SkillStore  # noqa: E402
from jevmaster.record.experiment import (  # noqa: E402
    aggregate_final_metrics,
    generate_exp_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# The official ALFWorld lists (frozen by scripts/make_alfworld_official_lists.py).
# They use the same item_id schema as every other AgentEval list.
ALFWORLD_ITEM_FILES = {
    "train": "data/agenteval/alfworld_train200.json",
    "id": "data/agenteval/alfworld_valid_seen.json",
    "ood": "data/agenteval/alfworld_valid_unseen.json",
}

# MiniGrid / BabyAI mixed split (frozen by scripts/make_minigrid_split.py):
# train 200 MixedTrainLocal -> eval-id 150 new seeds (same task-kind mix) +
# eval-ood 150 MixedTestLocal (held-out color-object combinations). Manifest
# records carry the env id and reset seed.
MINIGRID_ITEM_FILES = {
    "train": "data/minigrid/splits/minigrid_train.json",
    "id": "data/minigrid/splits/minigrid_id.json",
    "ood": "data/minigrid/splits/minigrid_ood.json",
}

# TextCraft-Synth official split (frozen by
# scripts/make_textcraft_synth_split.py): train 200 / ID 150 at recipe depth
# 2-4 (disjoint task instances), OOD 150 at depth 5-6. Records carry the
# target items and starting inventory; the world is
# data/textcraft_synth/recipes.json.
TEXTCRAFT_SYNTH_ITEM_FILES = {
    "train": "data/textcraft_synth/splits/textcraft_synth_train.json",
    "id": "data/textcraft_synth/splits/textcraft_synth_id.json",
    "ood": "data/textcraft_synth/splits/textcraft_synth_ood.json",
}

# Phase plans, one entry per phase:
#   * 2-tuple (tag, selector): the legacy AgentEval form. The selector picks
#     the vendored item list (train/test) when the env has one, else
#     ``env.params.split``. run.phase / tracker prefix = tag.
#   * 4-tuple (tag, phase_kind, prefix, params): explicit. ``phase_kind`` is
#     what run.phase records (train|eval),
#     ``prefix`` namespaces the tracker metrics, and ``params`` are
#     ``env.params.*`` overrides. Any benchmark with train / ID / OOD
#     manifests uses ``split_phase_plan`` (jevmaster/envs/splits.py), so the
#     phase names, metric namespaces and skill-snapshot behaviour are the
#     same for every dataset.
PHASE_PLANS = {
    # The AgentEval envs: train (frozen train items) -> eval (the vendored
    # AgentEval test set; skills still accumulate). One skill store carries
    # across both phases.
    "alfworld": split_phase_plan(ALFWORLD_ITEM_FILES),
    # MiniGrid: MixedTrainLocal train -> new-seed ID + MixedTestLocal OOD.
    "minigrid": split_phase_plan(MINIGRID_ITEM_FILES),
    # TextCraft-Synth: depth 2-4 train -> depth 2-4 new-instance ID + depth
    # 5-6 OOD, both eval phases starting from the post-train skill snapshot.
    "textcraft_synth": split_phase_plan(TEXTCRAFT_SYNTH_ITEM_FILES),

}

# The M3-era protocol (`--split-profile agenteval`, the default): the vendored
# AgentEval lists, in file order -- train 100 -> eval (the AgentEval test set),
# one skill store across both phases, the eval phase still accumulating (no
# snapshot, no freeze). This is how the M3 batches ran, so their numbers and
# the batches' episode order are directly comparable. The ID/OOD plans above
# stay available with `--split-profile official`.
AGENTEVAL_PHASE_PLANS = {
    "alfworld": [("train", "train"), ("eval", "test")],
    "minigrid": [("train", "train"), ("eval", "test")],
}

# Episode round caps: the same numbers for both split profiles, so a run on
# the official manifests is comparable with a run on the AgentEval lists.
# ALFWorld and TextCraft-Synth use the AgentGym AgentEval protocol's 30
# rounds; MiniGrid and TextCraft-Synth need a larger cap:
#   * MiniGrid official: 70% of episodes (140/200 jev) ended exactly at 30,
#     while every success finished in a median of 9 steps; 60 gives the long
#     levels (Maze/ImpUnlock/Seq/PutNext, 64-128 env steps) room without the
#     4-6x cost of the old 150.
AGENTEVAL_STEP_LIMITS = {"alfworld": 30, "minigrid": 60,
                         "textcraft_synth": 75}
OFFICIAL_STEP_LIMITS = dict(AGENTEVAL_STEP_LIMITS)


def profile_step_limit(env_name: str, profile: str) -> int | None:
    """The episode round cap a profile implies (None: keep the config value)."""
    limits = OFFICIAL_STEP_LIMITS if profile == "official" else AGENTEVAL_STEP_LIMITS
    return limits.get(env_name)



def phase_specs(env_name: str,
                profile: str = "agenteval") -> list[tuple[str, str, str, dict]]:
    """Normalize a phase plan into (tag, phase_kind, prefix, params).

    ``profile="agenteval"`` uses the M3-era AgentEval train/test lists for the
    four aligned benchmarks (file order, one accumulating store); other
    environments, and ``profile="official"``, use the frozen split manifests.
    Reads the plans at call time so tests can monkeypatch them.
    """
    plan = PHASE_PLANS.get(env_name, [("all", None)])
    if profile == "agenteval" and env_name in AGENTEVAL_PHASE_PLANS:
        plan = AGENTEVAL_PHASE_PLANS[env_name]
    specs = []
    for entry in plan:
        if len(entry) >= 4:
            tag, phase_kind, prefix, params = entry[:4]
            specs.append((tag, phase_kind, prefix, dict(params or {})))
            continue
        tag, selector = entry[0], entry[1]
        params = {}
        if selector is not None:
            if env_name in ITEM_FILES:
                params["items_file"] = item_file(env_name, selector)
            else:
                params["split"] = selector
        specs.append((tag, tag, tag, params))
    return specs


def _skills_root(config) -> Path:
    """The SkillStore root (it appends <env> itself)."""
    return Path(config.get("skills.dir") or "skills_store")


def _snapshot_root(config, base_exp_id: str) -> Path:
    root = config.get("skills.snapshot_dir") or (REPO_ROOT / "runs" / "skills")
    return Path(root) / f"{base_exp_id}-snapshots"


def needs_skills(config) -> bool:
    """Whether this run writes a skill store (System-2 is on)."""
    return str(config.get("system2.backend") or "").lower() not in ("", "none")


def snapshot_train_skills(config, env_name: str, base_exp_id: str,
                          records_dir: Path | None = None) -> Path:
    """Freeze the post-train skill store for the eval phases.

    ``SkillStore(root, env)`` stores items under ``root/<env>``, so the
    snapshot preserves that subdirectory and each eval phase gets its own
    copy as its ``skills.dir``.
    """
    src = _skills_root(config) / env_name
    dst = _snapshot_root(config, base_exp_id) / "train"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        lifecycle = config.section("skills").get("lifecycle") or {}
        if lifecycle.get("snapshot_require_validated"):
            # The store decides which revisions are validated enough for eval;
            # fatal rules are copied alongside.
            store = SkillStore(
                _skills_root(config), env_name, lifecycle=lifecycle)
            store.export_snapshot(dst / env_name)
        else:
            shutil.copytree(src, dst / env_name)
    else:  # S2 off (jev): an empty store snapshot
        (dst / env_name).mkdir(parents=True, exist_ok=True)

def seed_eval_skills(config, env_name: str, base_exp_id: str, tag: str,
                     records_dir: Path) -> None:
    """Point an eval phase at its own copy of the post-train snapshot.

    Fresh phase (no episodes.jsonl yet): rebuild from the immutable train
    snapshot, so ID and OOD start from the same library and a rerun starts
    clean. Resumed phase: keep the store, preserving what that eval phase
    picked up so far.

    The copy stays writable by default (``skills.freeze: false``): System-2
    keeps assisting in every phase, and a skill developed on an ID or OOD
    episode is published there. ``skills.freeze: true`` makes the copy
    read-only, so an eval phase measures only what train developed.
    """
    snapshot = _snapshot_root(config, base_exp_id) / "train" / env_name
    target_root = _snapshot_root(config, base_exp_id) / tag
    fresh = not (records_dir / "episodes.jsonl").exists()
    if fresh and target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / env_name
    if fresh:
        if snapshot.exists():
            shutil.copytree(snapshot, target)
        else:
            target.mkdir(parents=True, exist_ok=True)
    config.set("skills.dir", str(target_root))
    # None means "auto": the isolated eval copies stay writable, so every
    # phase may keep developing skills (train is where most episodes are).
    # An explicit true/false wins.
    if config.get("skills.freeze") is None:
        config.set("skills.freeze", False)


def _print_phase_summary(tag: str, summaries, records_dir: Path) -> None:
    n = len(summaries)
    print(f"\n=== Phase: {tag} ===")
    if not n:
        print("no episodes")
        return
    successes = sum(1 for s in summaries if s["success"])
    avg = sum(s["score"] for s in summaries) / n
    avg_clipped = sum(max(0.0, s["score"]) for s in summaries) / n
    skills_added = sum(s.get("skills_added", 0) for s in summaries)
    skill_turns = sum(s.get("skill_turns", 0) for s in summaries)
    print(f"episodes:      {n}")
    print(f"success rate:  {successes}/{n}")
    print(f"avg score:     {avg:.3f}  (negatives clipped at 0: {avg_clipped:.3f})")
    if skill_turns:
        print(f"skill turns:   {skill_turns}")
    print(f"skills added:  {skills_added}")
    print(f"skills total:  {summaries[-1].get('skills_total', 0)}")
    totals = summaries[-1].get("s2_totals")
    try:  # final counters incl. the run-end drain (written by agent.close)
        totals = json.loads((records_dir / "summary.json").read_text()).get("s2_totals") or totals
    except (OSError, ValueError):
        pass
    if totals:
        # Dual: run-level System-2 counters (reviews arrive asynchronously).
        print(f"S2 requests:   {totals.get('requests', 0)} (uncertain steps)")
        print(f"S2 reviews:    {totals.get('reviews', 0)}  declined {totals.get('declined', 0)}  "
              f"developed {totals.get('developed', 0)} (published {totals.get('published', 0)})  "
              f"cooldown {totals.get('cooldown', 0)}  dropped {totals.get('dropped', 0)}  "
              f"suppressed {totals.get('suppressed', 0)}  frozen {totals.get('frozen', 0)}")
    print(f"records:       {records_dir}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Benchmark config, e.g. configs/alfworld.yaml")
    parser.add_argument("--base", default=None, help="Optional base config to merge under --config")
    parser.add_argument("--mode", default=None, choices=["dual", "jev", "llm", "llm_agent"])
    parser.add_argument("--s1-backend", default=None, choices=["jev", "llm", "stub"],
                        help="System-1 backend (llm: the Qwen chat decides -- the llm_agent ablation)")
    parser.add_argument("--s2-backend", default=None, choices=["llm", "stub", "none"])
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--variations", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episode cap per phase (default: the full item list)")
    parser.add_argument("--exp-id", default=None, help="Base experiment id (each phase suffixes its tag)")
    parser.add_argument("--split-profile", default="agenteval",
                        choices=["agenteval", "official"],
                        help="agenteval (default): the M3-era AgentEval train/test "
                             "lists in file order, one accumulating skill store "
                             "(train -> eval). official: the frozen split manifests "
                             "(train -> eval-id + eval-ood, frozen post-train store).")
    parser.add_argument("--only-phase", default=None,
                        help="Comma-separated phase tags to run, e.g. 'train' "
                             "(diagnostic arms: skip the eval phases)")
    parser.add_argument("--skills-dir", default=None,
                        help="Skill store root override (skills.dir); use a distinct dir per concurrent run")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging for every phase")
    parser.add_argument("--swanlab", action="store_true",
                        help="Enable SwanLab logging for every phase (China-accessible wandb alternative)")
    parser.add_argument("--wandb-group", default=None, help="Shared wandb group across phases")
    parser.add_argument("--tag", action="append", default=None,
                        help="Experiment tag (repeatable), e.g. --tag M0; lands in meta.json "
                             "(record.tags) and the wandb/swanlab run tags")
    parser.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                        help="Config override (repeatable), VALUE parsed as YAML, "
                             "e.g. --set skills.top_k=5")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    overrides = {
        "run.mode": args.mode,
        "system1.backend": args.s1_backend,
        "system2.backend": args.s2_backend,
        "env.params.max_tasks": args.tasks,
        "env.params.variations": args.variations,
        "skills.dir": args.skills_dir,
        "record.wandb.group": args.wandb_group,
    }
    if args.tag:
        overrides["record.tags"] = list(args.tag)
    if args.wandb:
        overrides["record.wandb.enabled"] = True
    if args.swanlab:
        overrides["record.swanlab.enabled"] = True
    if args.wandb and args.swanlab:
        parser.error("--wandb and --swanlab are mutually exclusive")
    # Boolean toggles: only override when the flag was actually passed as true.
    overrides = {k: v for k, v in overrides.items() if v is not None}

    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    config = load_config(args.config, base_path=args.base, overrides=overrides)
    env_name = config.get("env.name", "mock")
    # The profile fixes the episode round cap unless the caller set one
    # explicitly (--set env.step_limit=...). Without this the AgentEval
    # protocol silently runs MiniGrid failures to 150 steps instead of 20,
    # which changes both the score and the runtime by ~7x.
    if not any(str(item).partition("=")[0].strip() == "env.step_limit"
               for item in (args.set or [])):
        limit = profile_step_limit(env_name, args.split_profile)
        if limit is not None:
            config.set("env.step_limit", limit)
    mode = (config.get("run.mode") or "dual").lower()
    record_dir = Path(config.get("record.dir", "records"))
    base_exp_id = args.exp_id or generate_exp_id(mode, env_name)
    original_episodes = config.get("run.episodes")

    specs = phase_specs(env_name, args.split_profile)
    if args.only_phase:
        wanted = {p.strip() for p in args.only_phase.split(",") if p.strip()}
        specs = [s for s in specs if s[0] in wanted]
        if not specs:
            parser.error(f"--only-phase {args.only_phase!r} matches no phase")
    # Skill-freeze policy: None = auto (every phase may develop skills, eval
    # copies included), an explicit true/false is honoured for every phase.
    freeze_policy = config.get("skills.freeze")
    # Every mode runs the full plan (jev: System-2 off; llm_agent: the LLM
    # decides), so episodes pair one-to-one across modes.

    eval_dirs = []
    # Multiple eval phases (ALFWorld ID/OOD) must start from the same frozen
    # post-train store: isolate each one's copy. A single-eval env keeps the
    # existing shared-store behavior (train -> eval accumulates in place).
    isolate_skills = sum(1 for _, kind, _, _ in specs if kind != "train") > 1
    for tag, phase_kind, prefix, params in specs:
        exp_id = f"{base_exp_id}-{tag}"
        # Reset from the run-level policy: seed_eval_skills may turn auto
        # (None) into True for this phase; a later train phase must not
        # inherit the previous eval phase's freeze.
        config.set("skills.freeze", freeze_policy)
        config.set("run.episodes", args.episodes if args.episodes is not None else original_episodes)
        # Clear the previous phase's selector before applying this one, so an
        # items_file/split from the YAML or an earlier phase cannot leak in.
        config.set("env.params.items_file", None)
        config.set("env.params.split", None)
        for key, value in params.items():
            config.set(f"env.params.{key}", value)
        # Phase flag (run.phase): the train / eval label carried into records.
        config.set("run.phase", phase_kind)
        config.set("record.exp_id", exp_id)
        # ONE tracker run per PHASE (resume="allow" restores it after a
        # crash): each phase gets its own run with its own step axis from 0.
        # The prefix namespaces the metrics (train/, eval/, eval/id/,
        # eval/ood/), and the final report is final/<prefix>/*.
        for tracker in ("wandb", "swanlab"):
            config.set(f"record.{tracker}.run_id", exp_id)
            config.set(f"record.{tracker}.prefix", prefix)

        # With multiple eval phases, each starts from its own copy of the
        # post-train snapshot (ID and OOD see the same starting library).
        # jev mode disables System-2, so there is no store to isolate.
        use_skills = (isolate_skills
                      and str(config.get("run.mode") or "").lower() != "jev"
                      and needs_skills(config))
        if use_skills and phase_kind != "train":
            seed_eval_skills(config, env_name, base_exp_id, tag, record_dir / exp_id)

        agent = build_agent(config)
        try:
            summaries = agent.run()
        finally:
            agent.close()
        if use_skills and phase_kind == "train":
            snapshot_train_skills(config, env_name, base_exp_id, record_dir / exp_id)
        _print_phase_summary(tag, summaries, record_dir / exp_id)
        if phase_kind != "train":
            eval_dirs.append((tag, record_dir / exp_id))

    # The final report: every phase -- the same point each phase's tracker run
    # received from the recorder's close(). The eval phases are the headline
    # numbers; train shows what the warm-up itself reached.
    for tag, _, _, _ in specs:
        final = aggregate_final_metrics([record_dir / f"{base_exp_id}-{tag}"])
        if not final:
            continue
        print(f"\n=== Final ({tag}) ===")
        for key in sorted(final):
            value = final[key]
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    for tag, records in eval_dirs:
        print(f"\n########## {tag}: protocol report ##########")
        analyze_protocol.main([str(records)])

    print("\n=== Protocol run complete ===")
    print(f"base experiment id: {base_exp_id}")
    for tag, records in eval_dirs:
        print(f"  {tag:9s} records: {records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
