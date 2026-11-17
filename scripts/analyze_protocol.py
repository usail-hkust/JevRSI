#!/usr/bin/env python
"""Summarize experiment records under the AgentGym AgentEval protocol.

Usage:
    python scripts/analyze_protocol.py records/<exp_id>-all [records/<exp_id2>-all ...]

Multiple experiment directories are pooled (e.g. separate dual/jev/llm_agent
runs of the same protocol).

and success (score 100), overall and per task.

ALFWorld: binary success (won) overall plus a per-task-type breakdown over the
six canonical task types.

MiniGrid / TextCraft: 0/1 success (mission completed / goal crafted),
aggregated as a success percentage per task and overall, plus average steps.

(levels 23-26 train/ID, 27-30 OOD), so the OOD horizon effect is visible,
plus average steps. Run eval-id and eval-ood records separately (the
protocol report already does).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def load_episodes(record_dirs):
    """Yield (env_name, episode_summary) pooled across experiment dirs."""
    for record_dir in record_dirs:
        record_dir = Path(record_dir)
        meta_path = record_dir / "meta.json"
        summary_path = record_dir / "summary.json"
        if not meta_path.exists() or not summary_path.exists():
            print(f"warning: skipping {record_dir} (missing meta.json/summary.json)", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text())
        env_name = meta.get("env", "")
        summary = json.loads(summary_path.read_text())
        for episode in summary.get("episodes", []):
            yield record_dir.name, env_name, episode


def _fmt_mean_std(scores) -> str:
    if not scores:
        return "0.000 +- 0.000"
    m = mean(scores)
    s = pstdev(scores) if len(scores) > 1 else 0.0
    return f"{m:.3f} +- {s:.3f}"


def analyze_alfworld(episodes) -> dict:
    per_type = defaultdict(lambda: {"total": 0, "success": 0})
    total = successes = 0
    for episode in episodes:
        task_type = episode.get("task", "unknown")
        won = bool(episode.get("success")) or float(episode.get("score", 0.0)) > 0
        per_type[task_type]["total"] += 1
        total += 1
        if won:
            per_type[task_type]["success"] += 1
            successes += 1

    print("\n=== ALFWorld protocol report (strict pass@1) ===")
    rate = 100.0 * successes / total if total else 0.0
    print(f"overall  episodes={total}  success={successes} ({rate:.1f}%)")
    print("\n-- per task type --")
    for task_type in sorted(per_type):
        stat = per_type[task_type]
        trate = 100.0 * stat["success"] / stat["total"] if stat["total"] else 0.0
        print(f"{task_type:32s} episodes={stat['total']:3d} success={stat['success']:3d} ({trate:.1f}%)")
    return {
        "total": total,
        "success": successes,
        "success_rate": successes / total if total else 0.0,
        "per_task_type": {k: dict(v) for k, v in per_type.items()},
    }


def _analyze_progression(title: str, episodes, label_fn=None) -> dict:
    """BALROG scoring (collect_and_summarize_results): per-episode progression
    aggregated per task and overall as a percentage with standard error, plus
    average steps. `score` in our episode records IS the BALROG progression
    (see the minigrid adapter).

    ``label_fn`` can split the per-task breakdown further (the environment reports
    one row per Treasure Hunter level so the OOD horizon effect is visible).
    """
    label_fn = label_fn or (lambda episode: episode.get("task", "unknown"))
    per_task = defaultdict(list)   # label -> [(progression, success, steps)]
    progressions = []
    steps = []
    successes = 0
    for episode in episodes:
        progression = float(episode.get("score", 0.0) or 0.0)
        success = bool(episode.get("success"))
        step = int(episode.get("steps", 0) or 0)
        per_task[label_fn(episode)].append((progression, success, step))
        progressions.append(progression)
        steps.append(step)
        successes += 1 if success else 0

    n = len(progressions)
    print(f"\n=== {title} protocol report (BALROG progression) ===")
    report = {"episodes_played": n}

    def _pct_stats(values):
        m = mean(values)
        se = (pstdev(values) / len(values) ** 0.5) if len(values) > 1 else 0.0
        return 100.0 * m, 100.0 * se

    if n:
        avg_pct, se_pct = _pct_stats(progressions)
        rate = 100.0 * successes / n
        print(
            f"overall progression: {avg_pct:.2f}% ± {se_pct:.2f}%   "
            f"success {successes}/{n} ({rate:.1f}%)   "
            f"average steps: {mean(steps):.1f}"
        )
        report.update({
            "progression_percentage": avg_pct,
            "standard_error": se_pct,
            "success": successes,
            "success_rate": successes / n,
            "average_steps": mean(steps),
        })

    print("-- per task --")
    report["per_task"] = {}
    for task in sorted(per_task):
        pairs = per_task[task]
        values = [p for p, _, _ in pairs]
        task_successes = sum(1 for _, ok, _ in pairs if ok)
        avg_pct, se_pct = _pct_stats(values)
        avg_steps = mean([s for _, _, s in pairs])
        task_rate = 100.0 * task_successes / len(pairs)
        print(
            f"{task:40s} episodes={len(values):3d}  "
            f"success {task_successes:3d} ({task_rate:5.1f}%)  "
            f"progression {avg_pct:5.2f}% ± {se_pct:.2f}%  "
            f"steps {avg_steps:5.1f}"
        )
        report["per_task"][task] = {
            "progression_percentage": avg_pct,
            "standard_error": se_pct,
            "episodes_played": len(values),
            "success": task_successes,
            "success_rate": task_rate / 100.0,
            "average_steps": avg_steps,
        }
    return report


def analyze_minigrid(episodes) -> dict:
    return _analyze_progression("MiniGrid", episodes)


def analyze_textcraft(episodes) -> dict:
    # score is THREAD's 0/1 reward, so progression % == success rate.
    return _analyze_progression("TextCraft", episodes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", help="records/<exp_id> directories (pooled)")
    parser.add_argument("--json-out", default=None, help="Optional path for a JSON report")
    args = parser.parse_args(argv)

    by_env = defaultdict(list)
    for exp_id, env_name, episode in load_episodes(args.records):
        by_env[env_name].append(episode)

    if not by_env:
        print("No episodes found in the given record directories.", file=sys.stderr)
        return 1

    report = {}
    for env_name, episodes in by_env.items():
        if env_name == "alfworld":
            report[env_name] = analyze_alfworld(episodes)
        elif env_name == "minigrid":
            report[env_name] = analyze_minigrid(episodes)
        elif env_name == "textcraft":
            report[env_name] = analyze_textcraft(episodes)
        else:
            total = len(episodes)
            succ = sum(1 for e in episodes if e.get("success"))
            print(f"\n=== {env_name}: {succ}/{total} success ===")
            report[env_name] = {"total": total, "success": succ}

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
