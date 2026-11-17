"""Per-experiment recording: JSONL artifacts on disk + optional wandb logging.

Every run (CLI invocation) gets a unique experiment id and its own directory::

    records/<exp_id>/
      meta.json        -- mode, env, timestamps, full merged config
      steps.jsonl      -- one StepRecord per executed environment action
      messages.jsonl   -- every LLM message (ReAct turns and S2 review/chat logs)
      skills.jsonl     -- every skill created / revised during the experiment
      s2_results.jsonl -- dual: every System-2 result (review decision, publication)
                          as System-1 received it
      s2_messages.jsonl, patches.jsonl, outcomes.jsonl
                       -- dual: written by the System-2 process itself
      summary.json     -- written at close: per-episode summaries + totals

Experiment ids never collide between runs, so different experiments are never
mixed. Repeated runs of the same configuration can share a wandb ``group`` for
aggregation while keeping their individual ids.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from jevmaster import __version__
from jevmaster.types import StepRecord

logger = logging.getLogger(__name__)


def generate_exp_id(mode: str, env_name: str) -> str:
    """A unique, sortable experiment id: 20261016-153012-ab12cd-<mode>-<env>."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    uniq = uuid.uuid4().hex[:6]
    return f"{stamp}-{uniq}-{_slug(mode or 'run')}-{_slug(env_name or 'env')}"


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in text).strip("-") or "x"


class _JsonlWriter:
    """Thread-safe appending JSONL writer."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._fh = path.open("a", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def prefix_metric_keys(data: Dict[str, Any], prefix: Optional[str]) -> Dict[str, Any]:
    """Namespace metric keys under a phase prefix: step/x -> train/step/x."""
    if not prefix:
        return data
    return {f"{prefix}/{key}": value for key, value in data.items()}


def _jev_model(config) -> str:
    return str(config.get("system1.jev.model") or "d3")


def _llm_model(config) -> str:
    return str(config.get("system2.llm.model") or "llm")


def default_tracker_group(config, mode: str, env_name: str) -> str:
    """Group name encoding the experiment settings: dataset, mode, models.

    Shared by all online trackers (wandb / swanlab): repeated batches of the
    same experiment (same dataset + mode + models) aggregate under one group.
    The model names are read from the effective config, so a gateway Jev, a
    local StartLux of any size and the S2 model all appear without the caller
    naming them; ``record.group_prefix`` (e.g. the stage) is prepended.
    """
    s1 = str(config.get("system1.backend") or "").lower()
    s2 = str(config.get("system2.backend") or "").lower()
    models = []
    if mode in ("dual", "jev", "llm_agent") and s1 == "jev":
        models.append(_jev_model(config))
    elif mode in ("llm_agent", "llm") and s1 == "llm":
        models.append(_llm_model(config))
    if mode == "dual" and s2 == "llm":
        models.append(_llm_model(config))
    elif mode == "llm_agent" and s2 == "llm":
        models.append(f"s2:{_llm_model(config)}")
    tag = "+".join(models) if models else "stub"
    prefix = str(config.get("record.group_prefix") or "").strip()
    return f"{prefix}-{env_name}-{mode}-{tag}" if prefix else f"{env_name}-{mode}-{tag}"


class TokenLedger:
    """Per-model token usage and USD cost accounting.

    Pricing comes from ``record.cost.usd_per_m``: model -> USD per 1M tokens,
    either a scalar (same rate for input and output) or ``{input, output}``.
    Models without an entry price 0 (e.g. internal/unpriced deployments).

    The ledger books:
      * System-1 Jev calls (dual/jev modes) -- in/out split from the gateway
        usage when the backend reports it, else the whole ``s1_tokens`` as
        input (conservative approximation);
      * System-1 LLM calls (llm_agent mode) under the LLM model;
      * System-2 skill chats in dual / llm_agent mode (``reply.prompt_tokens``
        / ``reply.completion_tokens``).
    """

    def __init__(self, jev_model: str, llm_model: str, usd_per_m: Dict[str, Any]):
        self.jev_model = str(jev_model or "jev")
        self.llm_model = str(llm_model or "llm")
        self._pricing: Dict[str, Tuple[float, float]] = {}
        for name, rate in (usd_per_m or {}).items():
            if isinstance(rate, dict):
                self._pricing[str(name)] = (
                    float(rate.get("input", 0) or 0),
                    float(rate.get("output", 0) or 0),
                )
            else:
                r = float(rate or 0)
                self._pricing[str(name)] = (r, r)
        self._usage: Dict[str, Dict[str, int]] = {}

    def book(self, model: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        if not model or (not tokens_in and not tokens_out):
            return
        row = self._usage.setdefault(model, {"in": 0, "out": 0})
        row["in"] += int(tokens_in or 0)
        row["out"] += int(tokens_out or 0)

    def _rates(self, model: str) -> Tuple[float, float]:
        return self._pricing.get(model, (0.0, 0.0))

    def snapshot(self) -> Dict[str, Any]:
        """Cumulative per-model usage + cost, and the running total."""
        models: Dict[str, Dict[str, Any]] = {}
        total_usd = 0.0
        for model, row in self._usage.items():
            in_rate, out_rate = self._rates(model)
            usd = row["in"] / 1e6 * in_rate + row["out"] / 1e6 * out_rate
            models[model] = {"in": row["in"], "out": row["out"], "usd": round(usd, 9)}
            total_usd += usd
        return {"models": models, "total_usd": round(total_usd, 9)}


class WandbLogger:
    """Optional wandb integration (no-op when disabled or not installed).

    ONE wandb run per experiment: ``id``/``name`` = ``run_id`` (stable across
    restarts -- a resumed run ``resume``s the same run), ``group`` aggregates
    repeated runs of the same experiment settings (dataset + mode + models).
    Within the run, per-episode metrics are namespaced by the phase
    ``prefix`` (``all/episode/score`` ... under the single-pass protocol);
    ``summary`` writes the run-level ``final/*`` report UNPREFIXED (top-level
    keys, shown in the wandb Summary tab).
    """

    def __init__(
        self,
        enabled: bool,
        project: str,
        group: Optional[str],
        mode: str,
        exp_id: str,
        run_config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        dir: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        self.enabled = bool(enabled) and str(mode).lower() not in ("disabled", "off", "false")
        self.prefix = (prefix or "").strip() or None
        self._run = None
        if not self.enabled:
            return
        try:
            import wandb  # optional dependency
        except ImportError:
            logger.warning(
                "record.wandb.enabled=true but wandb is not installed; "
                "falling back to file-only recording (pip install wandb)"
            )
            self.enabled = False
            return
        # run_id makes the run stable across phases: the first phase creates
        # it, later phases resume it (resume="allow": create if missing).
        run_id = run_id or exp_id
        try:
            self._run = wandb.init(
                project=project or "jevmaster",
                id=run_id,
                name=run_id,
                group=group or None,
                resume="allow",
                mode=str(mode).lower(),
                config=run_config,
                tags=list(tags or []) + ([self.prefix] if self.prefix else []),
                dir=str(dir) if dir else None,  # keep run files inside the experiment dir
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("wandb.init failed (%s); continuing file-only", exc)
            self.enabled = False
            self._run = None

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log one point; wandb maintains the monotonic step counter."""
        if not self.enabled or self._run is None:
            return
        try:
            payload = prefix_metric_keys(data, self.prefix)
            self._run.log(payload, **({"step": step} if step is not None else {}))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("wandb.log failed: %s", exc)

    def summary(self, data: Dict[str, Any]) -> None:
        """Final run-level results, UNPREFIXED (keys carry their own
        namespace, e.g. ``final/...``; shown in the wandb Summary tab)."""
        if not self.enabled or self._run is None:
            return
        try:
            self._run.summary.update(dict(data))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("wandb summary.update failed: %s", exc)

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:  # pragma: no cover - defensive
                pass
            self._run = None
        self.enabled = False


class SwanLabLogger:
    """SwanLab mirror of :class:`WandbLogger` (same semantics, China-accessible).

    Useful after wandb.ai access from mainland China ends (Oct 30, 2026):
    identical run/group/prefix behavior -- one run per experiment (stable
    ``run_id``, resumed on restart), settings-based group, per-episode
    metrics namespaced by the phase ``prefix`` (``all/...`` under the
    single-pass protocol). ``summary`` logs the run-level ``final/*`` report
    as one UNPREFIXED trailing point (swanlab has no summary API). Modes:
    ``online`` (swanlab.cn cloud), ``local`` (offline files, view with
    ``swanlab watch``), ``disabled``.
    """

    def __init__(
        self,
        enabled: bool,
        project: str,
        group: Optional[str],
        mode: str,
        exp_id: str,
        run_config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        dir: Optional[Union[str, Path]] = None,
        run_id: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
        self.enabled = bool(enabled) and str(mode).lower() not in ("disabled", "off", "false")
        self.prefix = (prefix or "").strip() or None
        self._run = None
        if not self.enabled:
            return
        try:
            import swanlab  # optional dependency
        except ImportError:
            logger.warning(
                "record.swanlab.enabled=true but swanlab is not installed; "
                "falling back to file-only recording (pip install swanlab)"
            )
            self.enabled = False
            return
        run_id = run_id or exp_id
        common = dict(
            project=project or "jevmaster",
            group=group or None,
            reinit=True,  # same-process re-init after the previous phase
            mode=str(mode).lower(),
            config=run_config,
            tags=list(tags or []) + ([self.prefix] if self.prefix else []),
        )
        try:
            # Later protocol phases continue the same cloud run.
            self._run = swanlab.init(id=run_id, name=run_id, resume="allow", **common)
        except Exception as exc:
            # A run deleted on the cloud (e.g. cleaned up after a crash)
            # cannot be resumed ("[Disabled_Resource] 实验已被删除"): fall back
            # to creating a fresh cloud run under the same NAME (a new auto
            # id) instead of silently dropping the whole cloud mirror.
            # resume then defaults to "never", which forbids passing an id.
            logger.warning("swanlab.init resume failed (%s); starting a fresh cloud run", exc)
            try:
                self._run = swanlab.init(name=run_id, **common)
            except Exception as exc2:  # pragma: no cover - defensive
                logger.warning("swanlab.init failed (%s); continuing file-only", exc2)
                self.enabled = False
                self._run = None

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            payload = prefix_metric_keys(data, self.prefix)
            self._run.log(payload, **({"step": step} if step is not None else {}))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("swanlab.log failed: %s", exc)

    def summary(self, data: Dict[str, Any]) -> None:
        """swanlab has no summary API; log one final, UNPREFIXED point
        (keys carry their own namespace, e.g. ``final/...``)."""
        if not self.enabled or self._run is None:
            return
        try:
            self._run.log(dict(data))
        except Exception as exc:
            logger.warning("swanlab.log failed: %s", exc)

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:  # pragma: no cover - defensive
                pass
            self._run = None
        self.enabled = False


class ExperimentRecorder:
    """Writes all artifacts for one experiment under ``records/<exp_id>/``."""

    def __init__(self, root: Union[str, Path], exp_id: str, meta: Dict[str, Any]):
        self.exp_id = exp_id
        self.dir = Path(root) / exp_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.created = datetime.now(timezone.utc).isoformat()

        meta = dict(meta or {})
        meta.update({"exp_id": exp_id, "created": self.created, "jevmaster_version": __version__})
        (self.dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

        self._steps = _JsonlWriter(self.dir / "steps.jsonl")
        self._messages = _JsonlWriter(self.dir / "messages.jsonl")
        self._skills = _JsonlWriter(self.dir / "skills.jsonl")
        self._s2_results = None   # opened on the first System-2 result (dual only)
        # Per-episode checkpoint: one line per COMPLETED episode. On resume
        # (same exp_id restarted after a crash) this file is replayed so the
        # runners can skip finished episodes and summary.json stays complete.
        self._episodes_writer = _JsonlWriter(self.dir / "episodes.jsonl")
        self._tracker = None
        self._ledger = None
        # The protocol phase this recorder belongs to ("train"/"eval"; "" for
        # plain single-phase runs): qualifies the final/* summary keys so the
        # phases of one experiment report side by side on one tracker run.
        self._phase = str(((meta.get("config") or {}).get("run") or {}).get("phase") or "").lower()
        self._episodes: List[Dict[str, Any]] = []
        self._skills_written = 0
        self._load_existing_episodes()

    def _load_existing_episodes(self) -> None:
        """Replay episodes.jsonl written by a previous (crashed) run.

        Restores the completed-episode list, and (via
        ``completed_episode_ids``) lets the runners skip episodes
        that already finished. Steps of an episode that was mid-flight when
        the crash happened remain in steps.jsonl as a partial tail; that
        episode is NOT in episodes.jsonl and will be re-run.
        """
        path = self.dir / "episodes.jsonl"
        if not path.exists():
            return
        # A crash mid-write can leave a torn final line WITHOUT a newline;
        # appending after it would corrupt the next line too. Drop the torn
        # tail (rewrite) so the file stays clean for future appends.
        try:
            raw = path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                text = raw.decode("utf-8", errors="replace")
                good_lines = text.split("\n")[:-1]  # drop the torn fragment
                path.write_text("\n".join(good_lines) + "\n", encoding="utf-8")
        except OSError:
            logger.warning("failed to read %s; starting episode list empty", path)
            return
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn tail line from the crash
                if isinstance(row, dict) and row.get("episode_id"):
                    self._episodes.append(row)
        except OSError:
            logger.warning("failed to read %s; starting episode list empty", path)
            return
        if self._episodes:
            logger.info(
                "resuming %s: %d episode(s) already complete, skipping them",
                self.exp_id, len(self._episodes),
            )

    def completed_episode_ids(self) -> set:
        """Episode ids already finished in a previous (or this) run."""
        return {e["episode_id"] for e in self._episodes if e.get("episode_id")}

    def episode_summaries(self) -> List[Dict[str, Any]]:
        """All completed-episode summaries so far, including resumed ones."""
        return list(self._episodes)

    # ---- online tracker -----------------------------------------------------
    def attach_tracker(self, tracker) -> None:
        """Attach the optional online mirror (WandbLogger / SwanLabLogger)."""
        self._tracker = tracker

    def attach_ledger(self, ledger: TokenLedger) -> None:
        """Attach the token/cost ledger (see :class:`TokenLedger`)."""
        self._ledger = ledger

    def _wandb_log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._tracker is not None:
            self._tracker.log(data, step=step)

    # ---- artifacts --------------------------------------------------------
    def log_step(self, record: StepRecord) -> None:
        """One executed environment action. Written to steps.jsonl only.

        Step-level details (per-action reward, confidence, latency, tokens)
        stay in the local records; the online tracker is updated once per
        completed episode instead (see ``log_episode``).
        """
        extra = record.extra or {}
        tokens_in = int(extra.get("s1_tokens_in") or 0)
        tokens_out = int(extra.get("s1_tokens_out") or 0)
        if not (tokens_in or tokens_out) and record.s1_tokens:
            # Backend reported no split: book the whole call as input.
            tokens_in = int(record.s1_tokens)
        self.book_s1(record.s1_backend, tokens_in, tokens_out)
        self._steps.write(record.to_dict())

    def book_s1(self, backend: str, tokens_in: int, tokens_out: int) -> None:
        """System-1 usage into the cost ledger."""
        if self._ledger is None:
            return
        if backend == "jev":
            self._ledger.book(self._ledger.jev_model, tokens_in, tokens_out)
        elif backend == "llm":
            self._ledger.book(self._ledger.llm_model, tokens_in, tokens_out)

    def log_message(self, episode_id: str, role: str, content: str, **context: Any) -> None:
        """One LLM conversation message (kept even after the window trims it)."""
        row = {"episode_id": episode_id, "role": role, "content": content}
        row.update(context)
        self._messages.write(row)

    def log_s2_chat(self, episode_id: str, job_id: str, system: str, user: str, reply) -> None:
        """A complete System-2 skill-extraction chat (dual mode).

        Written to messages.jsonl; its token usage is booked into the cost
        ledger. Details stay local -- the tracker only sees episode summaries.
        """
        if self._ledger is not None:
            self._ledger.book(
                self._ledger.llm_model,
                int(getattr(reply, "prompt_tokens", 0) or 0),
                int(getattr(reply, "completion_tokens", 0) or 0),
            )
        self._messages.write(
            {
                "episode_id": episode_id,
                "job_id": job_id,
                "role": "system",
                "content": system,
                "prompt_tokens": getattr(reply, "prompt_tokens", 0),
            }
        )
        self._messages.write(
            {"episode_id": episode_id, "job_id": job_id, "role": "user", "content": user}
        )
        self._messages.write(
            {
                "episode_id": episode_id,
                "job_id": job_id,
                "role": "assistant",
                "content": getattr(reply, "text", ""),
                "latency_ms": getattr(reply, "latency_ms", 0.0),
                "completion_tokens": getattr(reply, "completion_tokens", 0),
            }
        )

    def log_s2_result(self, row: Dict[str, Any]) -> None:
        """One System-2 result received by System-1 (dual mode).

        Books the review's LLM usage into the cost ledger and records a
        published skill in skills.jsonl.
        """
        if self._s2_results is None:
            self._s2_results = _JsonlWriter(self.dir / "s2_results.jsonl")
        if self._ledger is not None and row.get("type") == "review":
            self._ledger.book(
                self._ledger.llm_model,
                int(row.get("prompt_tokens") or 0),
                int(row.get("completion_tokens") or 0),
            )
        markdown = row.get("markdown")
        if markdown:
            self._skills_written += 1
            self._skills.write({
                "episode_id": row.get("episode_id", ""), "name": row.get("skill"),
                "kind": row.get("kind", "procedure"),
                "source": "system2", "op": row.get("op"), "markdown": markdown,
            })
        self._s2_results.write({k: v for k, v in row.items() if k != "markdown"})

    def log_skill(self, episode_id: str, skill, source: str) -> None:
        """A skill created during the experiment (full markdown preserved)."""
        self._skills_written += 1
        self._skills.write(
            {
                "episode_id": episode_id,
                "name": skill.name,
                "source": source,
                "markdown": skill.to_markdown(),
            }
        )

    def log_episode(self, summary: Dict[str, Any]) -> None:
        """Per-episode summary; the only point the online tracker sees.

        One log call per completed episode (the task unit), carrying this
        episode's result plus the dataset-level running aggregates and the
        cumulative token/cost consumption. Step-level details stay in
        steps.jsonl / messages.jsonl locally.
        """
        if self._ledger is not None:
            snap = self._ledger.snapshot()
            summary.setdefault("tokens", snap["models"])
            summary.setdefault("cost_usd", snap["total_usd"])
        # Checkpoint FIRST (flushed by _JsonlWriter.write): once the line
        # reaches the OS the episode counts as done, and a later crash
        # (SIGKILL included) will skip it on resume.
        try:
            self._episodes_writer.write(summary)
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to checkpoint episode %s", summary.get("episode_id"))
        self._episodes.append(summary)
        n = len(self._episodes)
        avg_score = sum(e.get("score", 0.0) for e in self._episodes) / n
        # Negative scores (e.g. an episode-ending failure) clipped at 0: the
        # progress-only mean, comparable across environments' score ranges.
        avg_clipped = sum(max(0.0, e.get("score", 0.0)) for e in self._episodes) / n
        successes = sum(1 for e in self._episodes if e.get("success"))
        payload = {
            # this episode
            "episode/score": summary.get("score", 0.0),
            "episode/steps": summary.get("steps", 0),
            "episode/success": 1.0 if summary.get("success") else 0.0,
            "episode/skills_added": summary.get("skills_added", 0),
            "episode/cumulative_reward": summary.get("cumulative_reward", 0.0),
            # dataset-level running results
            "dataset/episodes": float(n),
            "dataset/avg_score": avg_score,
            "dataset/avg_score_clipped": avg_clipped,
            "dataset/success_rate": successes / n,
        }
        if summary.get("mode") == "dual":
            # s2_triggers: reviews that reached development during this episode
            # (System-2 is asynchronous; credit can arrive from an earlier one).
            payload["episode/s2_triggers"] = summary.get("s2_triggers", 0)
            for key in ("s2_requests", "s2_declined", "s2_published", "s2_rejected",
                        "s2_errors", "s2_dropped", "skills_blocked", "s1_latency_ms"):
                if key in summary:
                    payload[f"episode/{key}"] = summary[key]
            # State memory: was a state skill pinned for this episode.
            payload["episode/state_program_used"] = 1.0 if summary.get("state_program") else 0.0
            for key in ("procedures_total", "states_total", "blocked_total"):
                if key in summary:
                    payload[f"dataset/{key}"] = float(summary[key])
            for key, value in (summary.get("s2_totals") or {}).items():
                payload[f"dataset/s2_{key}"] = float(value)
        # Skill-library growth is tracked for every skill-accumulating mode
        # (dual / llm_agent via the S2 worker); jev-only never reports the key.
        if "skills_total" in summary:
            payload["dataset/skills_total"] = float(summary.get("skills_total") or 0)
        if "skill_turns" in summary:
            payload["episode/skill_turns"] = summary["skill_turns"]
        if self._ledger is not None:
            snap = self._ledger.snapshot()
            for model, row in snap["models"].items():
                slug = model.replace("/", "_")
                payload[f"tokens/{slug}/in"] = row["in"]
                payload[f"tokens/{slug}/out"] = row["out"]
            payload["cost/total_usd"] = snap["total_usd"]
        # Phase-local, 0-based episode step: every protocol phase opens its
        # own tracker run, so each metric namespace starts at step 0.
        self._wandb_log(payload, step=n - 1)

    def close(self, final: Optional[Dict[str, Any]] = None) -> None:
        """Write summary.json and finish the tracker.

        Under the two-phase protocol both phases share ONE tracker run:
        this phase's finish() closes it, and the next phase's recorder
        re-attaches (swanlab: reinit=True + resume="allow", verified to
        continue the same cloud experiment; wandb: resume="allow" natively).
        """
        payload = {"exp_id": self.exp_id, "created": self.created, "episodes": self._episodes}
        if self._ledger is not None:
            snap = self._ledger.snapshot()
            payload["tokens"] = snap["models"]
            payload["cost_usd"] = snap["total_usd"]
        if final:
            payload.update(final)
        try:
            (self.dir / "summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to write summary.json")
        for writer in (self._steps, self._messages, self._skills, self._episodes_writer,
                       self._s2_results):
            if writer is None:
                continue
            try:
                writer.close()
            except Exception:  # pragma: no cover - defensive
                pass
        if self._tracker is not None:
            if self._episodes:
                # The run-level final/* report comes from the same
                # aggregation the offline analyzer uses (summary.json was
                # just written), so the tracker point and any later offline
                # aggregation are identical by construction.
                try:
                    final_metrics = aggregate_final_metrics([self.dir])
                    # Namespace the phase's final report by its tracker prefix
                    # (eval/id, eval/ood, train, eval), so phases of one
                    # experiment report side by side. Plain single-phase runs
                    # (no prefix, or the generic "all") keep final/*.
                    prefix = getattr(self._tracker, "prefix", None) or self._phase
                    if final_metrics and prefix and prefix != "all":
                        final_metrics = {
                            f"final/{prefix}/{k[len('final/'):]}": v
                            for k, v in final_metrics.items()
                        }
                    if final_metrics:
                        self._tracker.summary(final_metrics)
                except Exception:  # pragma: no cover - defensive
                    logger.warning("tracker summary failed", exc_info=True)
            self._tracker.finish()

    # ---- context manager ----------------------------------------------------
    def __enter__(self) -> "ExperimentRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _tracker_cfg(record_cfg):
    """(tracker section, logger class): swanlab when enabled, else wandb."""
    swanlab_cfg = record_cfg.get("swanlab", {}) or {}
    wandb_cfg = record_cfg.get("wandb", {}) or {}
    if swanlab_cfg.get("enabled"):
        return swanlab_cfg, SwanLabLogger
    return wandb_cfg, WandbLogger


def _build_tracker(config, record_cfg, mode: str, env_name: str, run_id: str,
                   prefix: Optional[str] = None, run_dir=None):
    """The online tracker of one run (swanlab when enabled, else wandb)."""
    tracker_cfg, logger_cls = _tracker_cfg(record_cfg)
    return logger_cls(
        enabled=bool(tracker_cfg.get("enabled", False)),
        project=tracker_cfg.get("project", "jevmaster"),
        # explicit group wins; default groups by experiment settings
        group=tracker_cfg.get("group") or default_tracker_group(config, mode, env_name),
        mode=str(tracker_cfg.get("mode", "online")),
        exp_id=run_id,
        run_config={
            "mode": mode,
            "env": env_name,
            **{k: v for k, v in (config.data or {}).items() if k != "record"},
        },
        tags=[mode, env_name] + [str(t) for t in (record_cfg.get("tags") or [])],
        dir=run_dir,
        run_id=run_id,
        prefix=prefix,
    )


# Run-level System-2 counters aggregated into the experiment's final report.
FINAL_S2_KEYS = (
    "requests", "reviews", "declined", "published", "rejected", "empty",
    "frozen", "errors", "dropped", "blocked",
)


def _cumulative_from_snapshots(snapshots: List[float], final: float = 0.0) -> float:
    """True total of a monotone counter from per-episode cumulative snapshots.

    ``snapshots`` are the counter's cumulative values as of each COMPLETED
    episode, in order -- monotone within one process. A drop marks a resume
    boundary (a restarted process restarts the counter from 0), so the true
    total is the sum of each monotone segment's last value. ``final`` is the
    last process's summary-level value (written at clean close; it can exceed
    the last snapshot when work happened after the final episode, e.g. the
    run-end System-2 drain), so the last segment takes the max of the two.
    """
    if not snapshots:
        return float(final or 0.0)
    total = 0.0
    last = snapshots[0]
    for s in snapshots[1:]:
        if s < last:          # drop -> a restarted process began here
            total += last
            last = s
        else:
            last = s
    return total + max(last, float(final or 0.0))


def aggregate_final_metrics(phase_dirs) -> Dict[str, Any]:
    """``final/*`` metrics over EVERY episode of the given record dirs.

    Single-pass protocol: usually one dir; the signature stays a list so a
    legacy multi-phase experiment still aggregates. Per-process counters
    (System-2 totals, the token/cost ledger) restart on a resumed run, so
    they are RECONSTRUCTED from the per-episode cumulative snapshots (see
    ``_cumulative_from_snapshots``) instead of trusting the last process's
    summary alone. The skill-library sizes are maxima (the library carries
    over and keeps growing).
    Dirs without a summary (not run / crashed before close) are skipped.
    """
    episodes: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {}
    tokens: Dict[str, Dict[str, float]] = {}
    skills_total = procedures_total = states_total = blocked_total = 0
    cost_usd = 0.0
    for d in phase_dirs:
        try:
            summary = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        eps = summary.get("episodes") or []
        episodes.extend(eps)
        # System-2 totals: reconstruct across resume boundaries.
        summary_totals = summary.get("s2_totals") or {}
        for key in FINAL_S2_KEYS:
            snaps = [float((e.get("s2_totals") or {}).get(key) or 0.0) for e in eps]
            if key in summary_totals or any(snaps):
                totals[key] = totals.get(key, 0.0) + _cumulative_from_snapshots(
                    snaps, summary_totals.get(key) or 0.0
                )
        # Token / cost ledger: same reconstruction, per model and in total.
        summary_tokens = summary.get("tokens") or {}
        models = set(summary_tokens) | {m for e in eps for m in (e.get("tokens") or {})}
        for model in models:
            row = tokens.setdefault(model, {"in": 0.0, "out": 0.0})
            srow = summary_tokens.get(model) or {}
            for side in ("in", "out"):
                snaps = [
                    float((e.get("tokens") or {}).get(model, {}).get(side) or 0.0)
                    for e in eps
                ]
                row[side] += _cumulative_from_snapshots(snaps, srow.get(side) or 0.0)
        cost_snaps = [float(e.get("cost_usd") or 0.0) for e in eps]
        cost_usd += _cumulative_from_snapshots(cost_snaps, summary.get("cost_usd") or 0.0)
        skills_total = max(skills_total, int(summary.get("skills_total") or 0))
    if not episodes:
        return {}
    for e in episodes:
        procedures_total = max(procedures_total, int(e.get("procedures_total") or 0))
        states_total = max(states_total, int(e.get("states_total") or 0))
        blocked_total = max(blocked_total, int(e.get("blocked_total") or 0))
    n = len(episodes)
    out: Dict[str, Any] = {
        "final/episodes": n,
        "final/success_rate": sum(1 for e in episodes if e.get("success")) / n,
        "final/avg_score": sum(float(e.get("score", 0.0)) for e in episodes) / n,
        # negatives clipped at 0: the progress-only mean, comparable across
        # environments' score ranges (same convention as dataset/avg_score_clipped)
        "final/avg_score_clipped": sum(max(0.0, float(e.get("score", 0.0))) for e in episodes) / n,
        "final/score_neg100": sum(1 for e in episodes if float(e.get("score", 0.0)) <= -99.0),
        "final/avg_steps": sum(float(e.get("steps", 0)) for e in episodes) / n,
        "final/skills_total": skills_total,
        "final/procedures_total": procedures_total,
        "final/states_total": states_total,
        "final/blocked_total": blocked_total,
        "final/cost_usd": round(cost_usd, 9),
    }
    for model, row in sorted(tokens.items()):
        slug = model.replace("/", "_")
        out[f"final/tokens_{slug}_in"] = row["in"]
        out[f"final/tokens_{slug}_out"] = row["out"]
    for key in FINAL_S2_KEYS:
        if key in totals:
            out[f"final/s2_{key}"] = totals[key]

    # ---- S1/S2 interaction rates -----------------------------------------
    # Derived from the same episodes/counters: emitted only when the phase
    # actually carries the data (a baseline with System-2 off does not grow
    # a row of misleading zeros).
    total_steps = sum(int(e.get("steps") or 0) for e in episodes)
    if total_steps:
        latency = sum(
            float(e.get("s1_latency_ms") or 0.0) * int(e.get("steps") or 0)
            for e in episodes
        )
        if latency:
            # Step-weighted mean: per-episode values are per-step averages.
            out["final/avg_s1_latency_ms"] = latency / total_steps
    # System-2 off (jev) still writes a zeroed s2_totals block;
    # no non-zero counter means there was no S1/S2 interaction to report.
    s2_active = any(float(v or 0) for v in totals.values())
    requests = totals.get("requests", 0)
    reviews = totals.get("reviews", 0)
    published = totals.get("published", 0)
    if s2_active and total_steps and "requests" in totals:
        out["final/s2_request_rate"] = requests / total_steps
    if s2_active and requests and "reviews" in totals:
        out["final/s2_review_rate"] = reviews / requests
    if s2_active and reviews and "declined" in totals:
        out["final/s2_decline_rate"] = totals["declined"] / reviews
    if s2_active and reviews and "published" in totals:
        out["final/s2_publish_rate"] = published / reviews

    if any(e.get("state_program") for e in episodes):
        out["final/state_used_rate"] = (
            sum(1 for e in episodes if e.get("state_program")) / n
        )
    skills_added = sum(int(e.get("skills_added") or 0) for e in episodes)
    if skills_added or (s2_active and any(e.get("skills_added") is not None for e in episodes)):
        out["final/skills_added"] = skills_added
    return out



def build_recorder(config, mode: str, env_name: str) -> ExperimentRecorder:
    """Create the recorder for one run from the ``record`` config section."""
    record_cfg = config.section("record")
    exp_id = record_cfg.get("exp_id") or generate_exp_id(mode, env_name)
    recorder = ExperimentRecorder(
        record_cfg.get("dir") or "records",
        exp_id,
        meta={"mode": mode, "env": env_name, "config": config.data},
    )
    # Online mirror: swanlab (China-accessible) takes precedence when enabled,
    # otherwise wandb. Both share the same settings-based group and the
    # protocol-driven run_id / phase prefix.
    tracker_cfg, _ = _tracker_cfg(record_cfg)
    recorder.attach_tracker(
        _build_tracker(
            config, record_cfg, mode, env_name,
            # protocol runs set record.<tracker>.run_id / .prefix per phase:
            # one run per phase + a train/ / eval_id/ ... metric namespace
            run_id=tracker_cfg.get("run_id") or exp_id,
            prefix=tracker_cfg.get("prefix") or None,
            run_dir=recorder.dir,
        )
    )
    # Token/cost ledger: per-model usage priced by record.cost.usd_per_m
    # (jev: gateway usage; llm_agent / System-2 chats: API usage fields).
    cost_cfg = record_cfg.get("cost", {}) or {}
    recorder.attach_ledger(
        TokenLedger(
            jev_model=_jev_model(config),
            llm_model=_llm_model(config),
            usd_per_m=cost_cfg.get("usd_per_m") or {},
        )
    )
    return recorder
