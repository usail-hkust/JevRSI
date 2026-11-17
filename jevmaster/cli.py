"""Command-line entrypoint: run the dual-system agent over a benchmark."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

from jevmaster.agent import build_agent
from jevmaster.config import load_config


def _build_overrides(args) -> Dict[str, object]:
    mapping = {
        "run.mode": args.mode,
        "env.name": args.env,
        "system1.backend": args.s1_backend,
        "system2.backend": args.s2_backend,
        "run.episodes": args.episodes,
        "env.params.max_tasks": args.tasks,
        "env.params.variations": args.variations,
        "env.params.split": args.split,
        "trigger.confidence_threshold": args.confidence_threshold,
        "record.exp_id": args.exp_id,
        "record.wandb.group": args.wandb_group,
    }
    if args.wandb:
        mapping["record.wandb.enabled"] = True
    return {k: v for k, v in mapping.items() if v is not None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dual-system (Jev + LLM) embodied agent")
    parser.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/alfworld.yaml)")
    parser.add_argument("--base", default=None, help="Optional base config to merge under --config")
    parser.add_argument("--mode", default=None, choices=["dual", "jev", "llm_agent"],
                        help="Agent mode (same loop, one ablation each): dual (default) | "
                             "jev (System-2 off: no skills / state memory) | "
                             "llm_agent (the LLM decides instead of Jev)")
    parser.add_argument("--env", default=None, help="Override env name")
    parser.add_argument("--s1-backend", default=None, choices=["jev", "llm", "stub"], help="System-1 backend")
    parser.add_argument("--s2-backend", default=None, choices=["llm", "stub", "none"], help="System-2 backend")
    parser.add_argument("--tasks", type=int, default=None, help="Max number of tasks")
    parser.add_argument("--variations", type=int, default=None, help="Variations per task")
    parser.add_argument("--split", default=None,
                        help="Data split: minigrid train|dev|test|both. "
                             "ALFWorld selects games via env.params.items_file "
                             "(official lists: alfworld_train200 / alfworld_valid_seen / "
                             "alfworld_valid_unseen), not --split")
    parser.add_argument("--episodes", type=int, default=None, help="Max episodes to run")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="System-2 trigger threshold")
    parser.add_argument("--exp-id", default=None, help="Experiment id (default: auto-generated timestamp-uniq-mode-env)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging (record.wandb.enabled)")
    parser.add_argument("--wandb-group", default=None, help="wandb group for same-config repeated experiments")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config, base_path=args.base, overrides=_build_overrides(args))

    agent = build_agent(config)
    try:
        summaries = agent.run()
    finally:
        agent.close()

    n = len(summaries)
    successes = sum(1 for s in summaries if s["success"])
    avg_score = sum(s["score"] for s in summaries) / n if n else 0.0
    avg_clipped = sum(max(0.0, s["score"]) for s in summaries) / n if n else 0.0
    total_triggers = sum(s.get("s2_triggers", 0) for s in summaries)
    skills_added = sum(s.get("skills_added", 0) for s in summaries)
    mode = summaries[0].get("mode", "dual") if summaries else (args.mode or "dual")
    exp_id = summaries[0].get("exp_id", args.exp_id or "") if summaries else (args.exp_id or "")
    print("\n=== Run summary ===")
    print(f"mode:          {mode}")
    print(f"experiment:    {exp_id}")
    print(f"episodes:      {n}")
    print(f"success rate:  {successes}/{n}")
    print(f"avg score:     {avg_score:.3f}  (negatives clipped at 0: {avg_clipped:.3f})")
    if mode == "jev":
        print("S2:            disabled (no skills)")
    elif getattr(agent, "s2_totals", None):
        t = agent.s2_totals
        print(f"S2 requests:   {t['requests']} (uncertain steps)")
        print(f"S2 reviews:    {t['reviews']}  declined {t['declined']}  "
              f"published {t['published']}  rejected {t['rejected']}  "
              f"errors {t['errors']}  dropped {t['dropped']}")
        print(f"blocked:       {t['blocked']} new skills this run")
    else:
        print(f"S2 triggers:   {total_triggers}")
    print(f"skills added:  {skills_added}")
    store = getattr(agent, "store", None)
    total = len(store) if store is not None else (summaries[-1].get("skills_total", 0) if summaries else 0)
    print(f"skills total:  {total}")
    if store is not None:
        print(f"  procedures {store.count('procedure')}  states {store.count('state')}  "
              f"blocked {store.blocked_count()}")
    print(f"records:       {Path(config.get('record.dir', 'records')) / (exp_id or '<exp_id>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
