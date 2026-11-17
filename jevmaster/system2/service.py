"""The System-2 process: asynchronous reviews and the only writer of skills.

System-1 emits step / episode_start / episode_end events and keeps acting.
A step below the confidence threshold opens a review job; the job waits
``after_steps`` further steps (or the episode end) so the judge sees what the
uncertain decision led to.  The judge may decline (NONE) or choose exactly one
component to write or revise: ``procedure`` or ``state``.

M5.3 world model (one per environment, shared by its tasks): every step carries
the check of the world model's prediction for the executed action.  At episode
end the live accuracy is credited to the model, and a train episode with a
wrong prediction (or an environment without a model yet) queues a world-model
job that skips the judge.  A written model is admitted on its accuracy over
recorded train transitions of every task.

At episode end the service credits the episode's outcome to every skill that
was shown or followed.  Three consecutive failed episodes block a skill from
retrieval; a revision resets the counters and unblocks it (System-2 is the
only writer, so this is race-free with System-1's reads).
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from jevmaster.skills.skill import KINDS, Skill
from jevmaster.skills.fatal_guard import FatalRuleTable
from jevmaster.skills.retrieval import tokenize
from jevmaster.skills.skill import slugify
from jevmaster.skills.state import SandboxError, StateGraph, StateProgram
from jevmaster.skills.store import SkillStore
from jevmaster.skills.world import (TransitionBuffer, WorldModel, actual, make_row,
                                    replay_sample, score)
from jevmaster.types import ReviewJob, ReviewResult

logger = logging.getLogger(__name__)
SHUTDOWN = {"type": "shutdown"}
_REPLAY_HISTORY = 30


@dataclass
class _Open:
    job_id: str
    uncertain: List[int] = field(default_factory=list)
    observation: str = ""
    inventory: str = ""
    candidates: List[str] = field(default_factory=list)


@dataclass
class _Episode:
    episode_id: str = ""
    task: str = ""
    goal: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    start: Dict[str, Any] = field(default_factory=dict)
    state_program: Optional[str] = None
    states: Dict[int, str] = field(default_factory=dict)
    open: Optional[_Open] = None


class _Files:
    def __init__(self, root: Optional[str]):
        from jevmaster.record.experiment import _JsonlWriter

        self.root = Path(root) if root else None
        self._writers: Dict[str, Any] = {}
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
            for name in ("s2_messages", "patches", "outcomes"):
                self._writers[name] = _JsonlWriter(Path(root) / f"{name}.jsonl")

    def write(self, name: str, row: Dict[str, Any]) -> None:
        writer = self._writers.get(name)
        if writer is not None:
            writer.write(row)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()


class S2Service:
    def __init__(self, system2, store: SkillStore, env_name: str,
                 emit: Callable[[Dict[str, Any]], None],
                 files: Optional[_Files] = None, after_steps: int = 3,
                 queue_size: int = 64, max_fixes: int = 1,
                 max_body_chars: int = 1200, max_program_chars: int = 6000,
                 step_budget_ms: float = 50.0, max_lines: int = 20000,
                 review_enabled: bool = True,
                 positive_only: bool = False, avoid_guard: bool = False,
                 state_require_informative: bool = False,
                 state_min_known_ratio: float = 0.25,
                 state_min_distinct: int = 1,
                 state_require_goal_numbers: bool = True,
                 prompt_features: Optional[Dict[str, Any]] = None,
                 fatal_admission: bool = False,
                 fatal_admission_min_support: int = 1,
                 judge_pool: int = 0,
                 dedup: bool = False, dedup_similarity: float = 0.6,

                 world_model: Optional[Dict[str, Any]] = None):
        self.system2 = system2
        self.store = store
        self.env_name = env_name
        self.emit = emit
        self.files = files or _Files(None)
        self.after_steps = max(0, int(after_steps))
        self.queue_size = max(1, int(queue_size))
        self.max_fixes = max(0, int(max_fixes))
        self.max_body_chars = max(0, int(max_body_chars))
        self.max_program_chars = int(max_program_chars)
        self.step_budget_ms = float(step_budget_ms)
        self.max_lines = int(max_lines)
        self.review_enabled = bool(review_enabled)
        self.positive_only = bool(positive_only)
        self.avoid_guard = bool(avoid_guard)
        self.state_require_informative = bool(state_require_informative)
        self.state_min_known_ratio = max(0.0, min(1.0, float(state_min_known_ratio)))
        self.state_min_distinct = max(1, int(state_min_distinct))
        self.state_require_goal_numbers = bool(state_require_goal_numbers)
        self.prompt_features = dict(prompt_features or {})
        self.fatal_admission = bool(fatal_admission)
        self.fatal_admission_min_support = max(1, int(fatal_admission_min_support))
        # M5.3: the judge sees the components retrieval ranks for this
        # situation, and a new component that re-states an existing one is
        # folded into it instead of adding a near-variant.
        self.judge_pool = max(0, int(judge_pool))
        self.dedup = bool(dedup)
        self.dedup_similarity = float(dedup_similarity)
        # The judge chooses among the component kinds that are enabled.
        self.judge_kinds = ["procedure", "state"]
        wm = dict(world_model or {})
        self.world_enabled = bool(wm.get("enabled", False))
        self.world_cfg = wm
        self._world_busy: set = set()
        self._world_lock = threading.Lock()
        self._world_scores: Dict[tuple, Dict[str, Any]] = {}
        self._episodes: Dict[str, _Episode] = {}
        self._ready: Deque[ReviewJob] = deque()
        self._cond = threading.Condition()
        self.closed = False

    # ---- events ----------------------------------------------------------
    def handle(self, event: Dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "episode_start":
            self._on_start(event)
        elif kind == "step":
            self._on_step(event)
        elif kind == "episode_end":
            self._on_end(event)

    def _episode(self, event: Dict[str, Any]) -> _Episode:
        ep = self._episodes.setdefault(event["episode_id"], _Episode())
        ep.episode_id = str(event.get("episode_id") or ep.episode_id)
        ep.task = event.get("task", ep.task)
        ep.goal = event.get("goal", ep.goal)
        return ep

    def _on_start(self, event: Dict[str, Any]) -> None:
        ep = self._episode(event)
        ep.start = {"goal": event.get("goal", ""),
                    "observation": event.get("observation", ""),
                    "inventory": event.get("inventory", "")}
        ep.state_program = event.get("state_program")

    def _on_step(self, event: Dict[str, Any]) -> None:
        ep = self._episode(event)
        ep.steps.append(event)
        step = int(event.get("step", 0))
        if event.get("state"):
            ep.states[step] = str(event["state"])
        if event.get("uncertain"):
            if ep.open is not None and step - ep.open.uncertain[0] >= 2 * max(self.after_steps, 1):
                self._release(ep)
            if ep.open is None:
                ep.open = _Open(job_id=event.get("job_id") or uuid.uuid4().hex[:12])
            ep.open.uncertain.append(step)
            ep.open.observation = event.get("observation", "")
            ep.open.inventory = event.get("inventory", "")
            ep.open.candidates = list(event.get("candidates") or [])
        elif ep.open is not None and step - ep.open.uncertain[-1] >= self.after_steps:
            self._release(ep)

    def _on_end(self, event: Dict[str, Any]) -> None:
        ep = self._episodes.pop(event["episode_id"], None) or _Episode()
        if ep.open is not None:
            self._release(ep, outcome=event)
        used = sorted(set(event.get("used_skills") or []))
        success = bool(event.get("success"))
        progress = float(event.get("progress", 1.0 if success else 0.0))
        ablation = bool(event.get("ablation"))
        baseline = None
        if self.store.lifecycle_mode == "relative":
            if ablation or not used:
                self.store.observe_baseline(ep.task, ep.goal, success, progress)
            baseline = self.store.baseline_for(ep.task, ep.goal)
        blocked = self.store.record_outcomes(used, success, baseline=baseline,
                                              progress=progress)
        if self.world_enabled and not self.store.frozen:
            self._world_after_episode(ep, event)
        self.files.write("outcomes", {
            "episode_id": event.get("episode_id", ""), "task": ep.task,
            "success": success, "progress": progress, "used": used,
            "ablation": ablation, "baseline": baseline,
            "gain_mean": (sum(self.store.get(n).gain for n in used
                              if self.store.get(n) is not None) / len(used)) if used else None,
            "blocked": [s.name for s in blocked],
        })
        if blocked:
            self.emit({"type": "blocked", "episode_id": event.get("episode_id", ""),
                       "skills": [s.name for s in blocked]})

    def _release(self, ep: _Episode, outcome: Optional[Dict[str, Any]] = None) -> None:
        op, ep.open = ep.open, None
        if op is None or not self.review_enabled:
            return
        shown: List[str] = []
        states: Dict[int, str] = {}
        for ev in ep.steps:
            step = int(ev.get("step", 0))
            if step in op.uncertain:
                states[step] = str(ev.get("state") or "")
                for name in ev.get("skills") or []:
                    if name not in shown:
                        shown.append(name)
        job = ReviewJob(
            job_id=op.job_id, env_name=self.env_name,
            episode_id=ep.episode_id,
            task=ep.task, goal=ep.goal,
            trajectory=[dict(s) for s in ep.steps],
            uncertain_steps=list(op.uncertain),
            observation=op.observation, inventory=op.inventory,
            candidates=op.candidates, outcome=dict(outcome) if outcome else None,
            shown_skills=shown, states=states,
            start=dict(ep.start), state_program=ep.state_program,
            state_errors=[],)
        with self._cond:
            if len(self._ready) >= self.queue_size:
                dropped = self._ready.popleft()
                self.emit({"type": "dropped", "job_id": dropped.job_id,
                           "episode_id": dropped.episode_id})
            self._ready.append(job)
            self._cond.notify()

    # ---- jobs ------------------------------------------------------------
    def next_job(self, timeout: float = 0.5) -> Optional[ReviewJob]:
        with self._cond:
            if not self._ready and not self.closed:
                self._cond.wait(timeout)
            return self._ready.popleft() if self._ready else None

    def finished(self) -> bool:
        with self._cond:
            return self.closed and not self._ready

    def run_job(self, job: ReviewJob) -> Dict[str, Any]:
        try:
            row = self._review(job)
        except Exception as exc:
            logger.warning("System-2 review %s failed: %s", job.job_id, exc, exc_info=True)
            row = {"type": "review", "job_id": job.job_id, "episode_id": job.episode_id,
                   "decision": None, "status": "error", "error": repr(exc)}
        finally:
            if job.kind_hint == "world_model":
                with self._world_lock:
                    self._world_busy.discard(_WORLD_KEY)
                for key in [k for k in list(self._world_scores) if k[0] == job.job_id]:
                    self._world_scores.pop(key, None)
        self.emit(row)
        return row

    # ---- world model -----------------------------------------------------
    def _world_target(self) -> Optional[Skill]:
        """The environment's world model (None: not written yet)."""
        models = self.store.snapshot(kind="world_model", include_blocked=True)
        if not models:
            return None
        return max(models, key=lambda s: ((s.hits + 1.0) / (s.checks + 2.0), s.name))

    def _world_after_episode(self, ep: _Episode, event: Dict[str, Any]) -> None:
        counts: Dict[tuple, List[int]] = {}
        for ev in ep.steps:
            name, ok = ev.get("world_model"), ev.get("prediction_ok")
            if name and ok is not None:
                c = counts.setdefault((name, ev.get("world_revision")), [0, 0])
                c[0] += 1
                c[1] += int(bool(ok))
        for (name, revision), (n, hits) in counts.items():
            self.store.record_predictions(name, n, hits, revision)
        if not self.review_enabled or event.get("ablation") or not ep.steps:
            return
        target = self._world_target()
        wrong = [ev for ev in ep.steps if ev.get("prediction_ok") is False
                 and (target is None or ev.get("world_model") == target.name)]
        if target is not None and not wrong:
            return
        with self._world_lock:
            if _WORLD_KEY in self._world_busy:
                return          # one model, one job at a time; its admission sees the buffers
            self._world_busy.add(_WORLD_KEY)
        job = ReviewJob(
            job_id=uuid.uuid4().hex[:12], env_name=self.env_name, episode_id=ep.episode_id,
            task=ep.task, goal=ep.goal, trajectory=[dict(s) for s in ep.steps],
            uncertain_steps=[], outcome=dict(event), start=dict(ep.start),
            kind_hint="world_model", target=target.name if target else None,
            misses=_pick_misses(wrong, int(self.world_cfg.get("max_misses", 8))),
            state_memory=self._state_transitions(ep))
        with self._cond:
            if len(self._ready) >= self.queue_size:
                dropped = self._ready.popleft()
                self.emit({"type": "dropped", "job_id": dropped.job_id,
                           "episode_id": dropped.episode_id})
                if dropped.kind_hint == "world_model":
                    with self._world_lock:
                        self._world_busy.discard(_WORLD_KEY)
            self._ready.append(job)
            self._cond.notify()

    def _state_transitions(self, ep: _Episode, limit: int = 30) -> str:
        """The episode's state memory as ``state --action--> state (count)`` lines,
        most frequent first (this task's graph of the pinned state program)."""
        name = ep.state_program or next(
            (ev.get("state_program") for ev in ep.steps if ev.get("state_program")), None)
        if not name:
            return ""
        graph = StateGraph.load(self.store.graph_dir(name) / f"{slugify(ep.task)}.json", name, ep.task)
        edges = sorted(graph.edges.values(), key=lambda e: (-int(e.get("count", 0)), e.get("action", "")))
        if not edges:
            return ""

        def label(node: str) -> str:
            return " ".join(str(graph.nodes.get(node, {}).get("label", node)).split())[:100]

        lines = [f"- {label(e.get('from', ''))} --{e.get('action', '')}--> {label(e.get('to', ''))} "
                 f"({int(e.get('count', 0))})" for e in edges[:limit]]
        if len(edges) > limit:
            lines.append(f"- ... ({len(edges) - limit} more transitions)")
        return f"state program {name}: {len(graph.nodes)} states, {len(edges)} transitions\n" + "\n".join(lines)

    def _compile_world(self, code: str) -> WorldModel:
        cfg = self.world_cfg
        return WorldModel.compile(code, int(cfg.get("max_program_chars", 20000)),
                                  float(cfg.get("budget_ms", 300)),
                                  int(cfg.get("max_lines", 500000)),
                                  int(cfg.get("effect_chars", 160)))

    def _world_replay(self, job: ReviewJob) -> tuple:
        """(rows, indices): recorded train transitions of every task (one block
        per task), then this episode.  Every step of the episode is scored, plus
        a sample spread over tasks and, within a task, over action verbs."""
        capacity = int(self.world_cfg.get("buffer_rows", 400))
        tasks = self.store.world_tasks()
        quota = max(5, int(self.world_cfg.get("replay_rows", 150)) // max(1, len(tasks)))
        rows: List[Dict[str, Any]] = []
        idx: List[int] = []
        for task in tasks:
            block = [r for r in TransitionBuffer(self.store.world_path(task), capacity).load()
                     if r.get("episode") != job.episode_id]
            idx += [len(rows) + i for i in replay_sample(block, quota, seed=f"{job.job_id}:{task}")]
            rows += block
        start = len(rows)
        rows += _episode_rows(job)
        return rows, idx + list(range(start, len(rows)))

    def _check_world(self, item: Skill, job: ReviewJob) -> Optional[str]:
        code = item.program_code()
        cache_key = (job.job_id, job.target, code)
        if cache_key in self._world_scores:
            return self._world_scores[cache_key].get("problem")
        problem = self._world_problem(code, job)
        self._world_scores[cache_key] = problem
        return problem.get("problem")

    def _world_problem(self, code: str, job: ReviewJob) -> Dict[str, Any]:
        cfg = self.world_cfg
        try:
            model = self._compile_world(code)
        except SandboxError as exc:
            return {"problem": str(exc)}
        rows, idx = self._world_replay(job)
        memory = int(cfg.get("memory", 200))
        new = score(model, rows, idx, memory)
        out: Dict[str, Any] = {"accuracy": new["accuracy"], "n": new["checked"]}
        if new["errors"]:
            out["problem"] = (f"predict() raised on {len(new['errors'])} of {new['n']} recorded "
                              f"steps, e.g.: {new['errors'][0][:300]}")
            return out
        if new["dumps"]:
            out["problem"] = ("an effect text is a dump of a dict or list; write what the action "
                              "does in plain words")
            return out
        min_cov = float(cfg.get("min_coverage", 0.5))
        if new["coverage"] < min_cov:
            out["problem"] = (
                f"only {new['checked']} of {new['n']} recorded steps got a checkable prediction "
                f"for the action that was executed (need {min_cov:.0%}). Predict every action "
                f"you have a rule for, with checkable fields.")
            return out
        old_acc = None
        if job.target and self.store.get(job.target) is not None:
            try:
                old = score(self._compile_world(self.store.get(job.target).program_code()),
                            rows, idx, memory)
                old_acc = old["accuracy"] if not old["errors"] else 0.0
            except SandboxError:
                old_acc = 0.0
        tol = float(cfg.get("regression_tolerance", 0.02))
        min_acc = float(cfg.get("min_accuracy", 0.6))
        if old_acc is not None and new["accuracy"] < old_acc - tol:
            head = (f"on {new['n']} recorded steps the revised model is right on "
                    f"{new['accuracy']:.0%} of its checked predictions, the current one on "
                    f"{old_acc:.0%}; the revision must not be less accurate.")
        elif old_acc is None and new["accuracy"] < min_acc:
            head = (f"on {new['n']} recorded steps the model is right on {new['accuracy']:.0%} "
                    f"({new['hits']}/{new['checked']}) of its checked predictions (need {min_acc:.0%}).")
        else:
            return out
        from jevmaster.prompts.system2 import render_miss

        examples = _pick_wrong(new["wrong"], 5)
        out["problem"] = (head + " Wrong predictions (predicted vs what happened):\n"
                          + "\n".join(render_miss(w) for w in examples))
        return out


    def close(self) -> None:
        for ep in list(self._episodes.values()):
            if ep.open is not None:
                self._release(ep)
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    # ---- review ----------------------------------------------------------
    def _blocks(self, names: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name in names:
            skill = self.store.get(name)
            if skill is not None:
                out[name] = _clip_block(skill.as_review_block(
                    show_gain=self.store.lifecycle_mode == "relative"))
        return out

    def _library_blocks(self, job: ReviewJob, include_blocked: bool) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for kind in self.judge_kinds:
            items = [s for s in self.store.snapshot(
                kind=kind, include_blocked=include_blocked, task=job.task or None)
                if s.blocked == include_blocked]
            for skill in items[:6]:
                out[skill.name] = _clip_block(skill.as_review_block(
                    show_gain=self.store.lifecycle_mode == "relative"))
        return out

    def _retrieval_pool(self, job: ReviewJob) -> Dict[str, str]:
        """The components that actually match this situation, best first.

        With ``system2.judge_pool`` > 0 the judge sees what retrieval ranks
        for the uncertain step instead of the six oldest components, which is
        what stops it from writing a near-variant of a skill it already has.
        """
        from jevmaster.skills.retrieval import BM25Retriever, retrieval_query
        from jevmaster.types import Observation

        limit = self.judge_pool
        if limit <= 0:
            return {}
        step = job.uncertain_steps[-1] if job.uncertain_steps else None
        state = str((job.states or {}).get(step, "")) if step is not None else ""
        obs = Observation(task=job.task, goal=job.goal,
                          text=job.observation, inventory=job.inventory)
        query = retrieval_query(obs, state)
        retriever = BM25Retriever(self.store, top_k=limit)
        out: Dict[str, str] = {}
        for kind in self.judge_kinds:
            ranked = [skill for _score, skill in retriever.query_scored(query, kind, job.task or None)
                      if not skill.blocked]
            for skill in ranked[:limit]:
                out[skill.name] = _clip_block(skill.as_review_block(
                    show_gain=self.store.lifecycle_mode == "relative"))
        return out

    def _nearest(self, item: Skill, target: Optional[str] = None) -> Optional[tuple]:
        """(similarity, skill) of the closest existing component of the same kind."""
        from jevmaster.skills.retrieval import tokenize

        def meta(s: Skill) -> set:
            return set(tokenize(" ".join([s.name or "", s.when or "",
                                          " ".join(s.keywords or [])])))

        mine = meta(item)
        if not mine:
            return None
        best = None
        for other in self.store.library():
            if other.kind != item.kind or other.name == item.name or other.name == target:
                continue
            theirs = meta(other)
            if not theirs:
                continue
            sim = len(mine & theirs) / len(mine | theirs)
            if best is None or sim > best[0]:
                best = (sim, other)
        return best

    def _fatal_advice(self, item: Skill, job: ReviewJob) -> Optional[str]:
        if not self.fatal_admission:
            return None
        path = self.store.dir / "_fatal_rules.json"
        if not path.exists():
            return None
        table = FatalRuleTable(path, frozen=True,
                               min_support=self.fatal_admission_min_support)
        actions: List[str] = []
        if item.kind == "procedure":
            for line in item.body.splitlines():
                for match in _ADVICE_ACTION_RE.finditer(line):
                    actions.append(match.group(0).strip())
        for action in actions:
            rule = table.matching_rule(job.task, action, job.goal)
            if rule is not None:
                return (f"recommends a learned fatal action {action[:80]!r} "
                        f"(object {', '.join(rule.get('tokens') or [])})")
        return None

    def _check(self, item: Skill, job: ReviewJob) -> Optional[str]:
        if item.kind not in KINDS:
            return f"unknown kind {item.kind!r}"
        if item.kind == "world_model":
            if not str(item.name or "").strip():
                return "missing name"
            return self._check_world(item, job)
        if not str(item.name or "").strip():
            return "missing name"
        if not str(item.body or "").strip():
            return "missing body"
        if item.kind == "procedure" and len(item.body) > self.max_body_chars:
            return f"body is {len(item.body)} chars (limit {self.max_body_chars})"
        if item.kind == "procedure":
            if self.positive_only and _NEGATION_RE.search(item.body):
                return ("write positive advice only; put forbidden actions/objects "
                        "in the structured avoid: list, not in the prose")
            if item.avoid:
                if len(item.avoid) > 5:
                    return "avoid accepts at most 5 patterns"
                corpus = " ".join(
                    [f"{ev.get('action', '')} {ev.get('feedback', '')}"
                     for ev in job.trajectory] + list(job.candidates))
                corpus_tokens = set(tokenize(corpus))
                for raw in item.avoid:
                    pat = str(raw).strip()
                    if not (3 <= len(pat) <= 80):
                        return "each avoid pattern must be 3-80 characters"
                    toks = set(tokenize(pat)) - set(tokenize(job.task or ""))
                    if not toks:
                        return "avoid patterns must name an action or object"
                    if _NEGATION_RE.search(pat):
                        return "avoid entries are patterns, not negated sentences"
                    if self.avoid_guard and not (toks & corpus_tokens):
                        return "avoid pattern is not grounded in the recorded episode"
        problem = self._fatal_advice(item, job)
        if problem is not None:
            return problem
        if item.kind == "procedure":
            return None
        # state: compile and replay this episode
        try:
            program = StateProgram.compile(item.program_code(), self.max_program_chars,
                                           self.step_budget_ms, self.max_lines)
        except SandboxError as exc:
            return str(exc)
        history: List[str] = []
        labels: List[str] = []
        data_values: List[Any] = []

        def run_state(goal, observation, inventory, hist):
            label, data = program(goal, observation, inventory, hist)
            labels.append(str(label))
            data_values.append(data)
            return label, data

        try:
            run_state(job.goal, job.start.get("observation", ""),
                      job.start.get("inventory", ""), history)
            for ev in job.trajectory:
                run_state(job.goal, ev.get("feedback", ""),
                          ev.get("next_inventory") or ev.get("inventory") or "",
                          history[-_REPLAY_HISTORY:])
                history.append(f"{ev.get('action', '')} -> {ev.get('feedback', '')}")
        except SandboxError as exc:
            return f"replay failed: {exc}"
        if self.state_require_informative:
            unknown = {"", "(unknown)", "unknown", "none", "null"}
            known = [label for label in labels if label.lower() not in unknown]
            if not known or len(known) / max(1, len(labels)) < self.state_min_known_ratio:
                return "state output is uninformative (all steps unknown/null)"
            if len(set(known)) < self.state_min_distinct:
                return "state output is constant across the replayed episode"
            if (self.state_require_goal_numbers and _goal_has_count(job.goal)
                    and not any(_has_positive_number(data) for data in data_values)):
                return "goal has an explicit count; state must compute remaining quantities"
        return None

    def _review(self, job: ReviewJob) -> Dict[str, Any]:
        if job.kind_hint == "world_model":
            result: ReviewResult = self.system2.review(
                job, {}, {}, lookup=self.store.get, blocked_blocks={},
                check=lambda item: self._check(item, job),
                max_fixes=int(self.world_cfg.get("max_fixes", 2)),
                features=self.prompt_features)
            return self._finish_review(job, result)
        existing = self._retrieval_pool(job)
        if not existing:
            existing = self._library_blocks(job, include_blocked=False)
        # A shown procedure already has its full text above; keep the library
        # list for the components not shown at the uncertain step.
        existing = {k: v for k, v in existing.items() if k not in job.shown_skills}
        blocked = self._library_blocks(job, include_blocked=True)
        blocked = {k: v for k, v in blocked.items() if k not in existing}
        result: ReviewResult = self.system2.review(
            job, self._blocks(job.shown_skills), existing,
            lookup=self.store.get, blocked_blocks=blocked,
            check=lambda item: self._check(item, job), max_fixes=self.max_fixes,
            features=self.prompt_features)
        return self._finish_review(job, result)

    def _finish_review(self, job: ReviewJob, result: ReviewResult) -> Dict[str, Any]:
        for chat in result.chats:
            reply = chat["reply"]
            self.files.write("s2_messages", {
                "job_id": job.job_id, "episode_id": job.episode_id,
                "turn": chat["turn"], "messages": chat["messages"],
                "reply": getattr(reply, "text", ""),
                "prompt_tokens": getattr(reply, "prompt_tokens", 0),
                "completion_tokens": getattr(reply, "completion_tokens", 0),
            })
        row: Dict[str, Any] = {
            "type": "review", "job_id": job.job_id, "episode_id": job.episode_id,
            "job": job.kind_hint or "review", "misses": len(job.misses),
            "uncertain_steps": job.uncertain_steps, "decision": result.decision,
            "target": result.target, "kind": result.kind, "reason": result.reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }
        if result.decision == "NONE":
            row["status"] = "declined"
        else:
            row.update(self._publish(job, result))
        self.files.write("patches", {k: v for k, v in row.items() if k != "markdown"})
        return row

    def _publish(self, job: ReviewJob, result: ReviewResult) -> Dict[str, Any]:
        if self.store.frozen:
            return {"status": "frozen"}
        if result.skill_markdown is None:
            return {"status": "empty"}
        try:
            skill = Skill.from_markdown(result.skill_markdown)
        except ValueError as exc:
            return {"status": "rejected", "rejected": f"invalid markdown: {exc}"}
        skill.env = self.env_name
        skill.source = "system2"
        skill.kind = result.kind
        skill.tasks = [job.task] if job.task else []
        skill.goals = [job.goal] if job.goal else []
        skill.trigger_context = (f"uncertain steps {job.uncertain_steps} "
                                 f"({job.episode_id or 'episode'})")
        world = skill.kind == "world_model"
        if world:
            skill.trigger_context = (f"world model: {len(job.misses)} wrong prediction(s) "
                                     f"({job.episode_id or 'episode'})")
            skill.tasks = []                  # one model serves every task
            current = self._world_target()
            if current is not None:
                skill.name = current.name     # the environment has one world model
        # The framework, never System-2, decides how a skill is applied.
        problem = self._check(skill, job)
        if problem is not None:
            return {"status": "rejected", "rejected": problem.splitlines()[0]}
        target = result.target
        if (target is None or world) and self.store.get(skill.name) is not None:
            target = skill.name              # NEW under an existing name: a revision
        duplicate = None
        if target is None and self.dedup and not world:
            near = self._nearest(skill)
            if near is not None and near[0] >= self.dedup_similarity:
                target = near[1].name        # same idea: update it, do not add a variant
                duplicate = {"near": near[1].name, "proposed": skill.name,
                             "similarity": round(near[0], 2)}
        if target is not None:
            existing = self.store.get(target)
            if existing is not None and existing.kind != skill.kind:
                return {"status": "rejected",
                        "rejected": f"{target} is a {existing.kind}, not a {skill.kind}"}
        out = self._store_publish(skill, target)
        if duplicate is not None:
            out["dedup"] = duplicate
        if world and out.get("status") == "published":
            replay = self._world_scores.pop((job.job_id, job.target, skill.program_code()), {}) or {}
            self.store.set_replay(out["skill"], replay.get("accuracy", 0.0), replay.get("n", 0))
            out["replay_accuracy"] = round(float(replay.get("accuracy", 0.0)), 4)
            out["replay_n"] = int(replay.get("n", 0))
        return out

    def _store_publish(self, skill: Skill, target: Optional[str]) -> Dict[str, Any]:
        if target is not None:
            published = self.store.revise(target, skill)
            if published is None:
                return {"status": "rejected", "rejected": f"unknown skill {target}"}
            op = "revise"
        else:
            published = self.store.add(skill)
            op = "add"
        return {"status": "published", "op": op, "skill": published.name,
                "kind": published.kind, "markdown": published.to_markdown()}


_WORLD_KEY = "world_model"        # the busy marker of the environment's one model


def _episode_rows(job: ReviewJob) -> List[Dict[str, Any]]:
    """The job's trajectory as transition rows (history as System-1 had it)."""
    rows: List[Dict[str, Any]] = []
    history: List[str] = []
    for ev in job.trajectory:
        rows.append(make_row(job.goal, ev.get("observation", ""), ev.get("inventory") or "",
                             history, ev.get("candidates") or [], ev.get("action", ""),
                             ev.get("feedback", ""), ev.get("next_inventory") or "",
                             ev.get("reward") or 0.0, bool(ev.get("done")), job.episode_id))
        history.append(f"{ev.get('action', '')} -> {' '.join(str(ev.get('feedback', '')).split())[:300]}")
    return rows


def _pick_wrong(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Up to ``limit`` items, one per action verb first (most recent first)."""
    items = list(reversed(items))
    picked: List[Dict[str, Any]] = []
    verbs: set = set()
    for item in items:
        verb = (str(item.get("action") or "").split() or [""])[0]
        if verb not in verbs:
            verbs.add(verb)
            picked.append(item)
    for item in items:
        if len(picked) >= limit:
            break
        if item not in picked:
            picked.append(item)
    return picked[:limit]


def _current_state(summary: Any) -> str:
    """The ``current:`` line of a state-memory summary ("" without one)."""
    for line in str(summary or "").splitlines():
        if line.startswith("current: "):
            return line[len("current: "):][:160]
    return ""


def _pick_misses(events: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Wrong live predictions as prompt rows: predicted vs what happened."""
    out = []
    for ev in events:
        row = make_row("", ev.get("observation", ""), ev.get("inventory") or "", [], [],
                       ev.get("action", ""), ev.get("feedback", ""),
                       ev.get("next_inventory") or "", ev.get("reward") or 0.0,
                       bool(ev.get("done")))
        out.append({"step": int(ev.get("step", 0)), "action": ev.get("action", ""),
                    "predicted": ev.get("prediction") or {},
                    "failed": list(ev.get("prediction_failed") or []),
                    "before": " ".join(str(ev.get("observation", "")).split())[:400],
                    "inventory": " ".join(str(ev.get("inventory") or "").split())[:150],
                    "state": _current_state(ev.get("state")),
                    "actual": actual(row)})
    return _pick_wrong(out, limit)



_ADVICE_ACTION_RE = re.compile(
    r"\b(?:focus on|go to|pick up|take|put|place|open|close|connect|"
    r"disconnect|use|move|drop)\b[^.\n;]*", re.IGNORECASE)


_COUNT_GOAL_RE = re.compile(
    r"\b\d+\s*x\b|\b(?:craft|collect|gather|make|mine)\s+\d+\b",
    re.IGNORECASE)


def _goal_has_count(goal: str) -> bool:
    return bool(_COUNT_GOAL_RE.search(str(goal or "")))


def _has_positive_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) > 0
    if isinstance(value, str):
        try:
            return float(value) > 0
        except ValueError:
            return False
    if isinstance(value, dict):
        return any(_has_positive_number(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_positive_number(v) for v in value)
    return False


_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|don't|never|avoid|must\s+not|should\s+not|"
    r"not\s+(?:focus|take|open|go|pick|use))\b", re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def _grounded(action: str, legal: List[str]) -> bool:
    """Exact or word-boundary prefix for a grounded action string."""
    a = _norm(action)
    if not a:
        return False
    for candidate in legal:
        c = _norm(candidate)
        if c == a:
            return True
        if len(a) >= 8 and c.startswith(a) and (len(c) == len(a) or c[len(a)] in " 0123456789"):
            return True
    return False


def _clip_block(text: str, limit: int = 1200) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---- process entry points -------------------------------------------------
def serve(config_data: Dict[str, Any], env_name: str, record_dir: Optional[str],
          events, results, log_level: int = logging.INFO) -> None:
    from jevmaster.config import Config
    from jevmaster.system2 import build_system2

    if not logging.getLogger().handlers:
        logging.basicConfig(level=log_level,
                            format="%(asctime)s %(levelname)s [s2] %(name)s: %(message)s")
    config = Config(config_data)
    skills_cfg = config.section("skills")
    proc_cfg = config.section("system2.process")
    state_cfg = skills_cfg.get("state") or {}
    store = SkillStore(skills_cfg.get("dir", "skills_store"), env_name,
                       frozen=bool(skills_cfg.get("freeze")),
                       block_after=int(skills_cfg.get("block_after", 3)),
                       lifecycle=skills_cfg.get("lifecycle") or {})
    files = _Files(record_dir)
    service = S2Service(
        build_system2(config),
        store,
        env_name, emit=results.put, files=files,
        judge_pool=int(config.get("system2.judge_pool", 0) or 0),
        dedup=bool(skills_cfg.get("dedup", False)),
        dedup_similarity=float(skills_cfg.get("dedup_similarity", 0.6)),
        world_model=skills_cfg.get("world_model") or {},
        after_steps=int(proc_cfg.get("after_steps", 3)),
        queue_size=int(proc_cfg.get("queue_size", 64)),
        max_fixes=int(proc_cfg.get("max_fixes", 1)),
        max_body_chars=int(skills_cfg.get("max_body_chars", 1200)),
        max_program_chars=int(state_cfg.get("max_program_chars", 6000)),
        step_budget_ms=float(state_cfg.get("step_budget_ms", 50)),
        max_lines=int(state_cfg.get("max_lines", 20000)),
        review_enabled=bool(config.get("system2.review", True)),
        positive_only=bool(skills_cfg.get("positive_only", False)),
        avoid_guard=bool(skills_cfg.get("avoid_guard", False)),
        state_require_informative=bool(state_cfg.get("require_informative", False)),
        state_min_known_ratio=float(state_cfg.get("min_known_ratio", 0.25)),
        state_min_distinct=int(state_cfg.get("min_distinct_states", 1)),
        state_require_goal_numbers=bool(state_cfg.get("require_goal_numbers", True)),
        fatal_admission=bool((skills_cfg.get("fatal_guard") or {}).get("admission", False)),
        fatal_admission_min_support=int((skills_cfg.get("fatal_guard") or {}).get(
            "admission_min_support", 1)),
        prompt_features={
            "positive_only": bool(skills_cfg.get("positive_only", False)),
            "avoid": bool(skills_cfg.get("avoid_guard", False)
                          or skills_cfg.get("positive_only", False)),
            "state_quality": bool(state_cfg.get("require_goal_match", False)
                                  or state_cfg.get("require_informative", False)
                                  or state_cfg.get("surface_facts", False)),
            "lifecycle": str((skills_cfg.get("lifecycle") or {}).get("mode", "absolute")) == "relative",
            "progress": bool(config.get("system2.progress_feedback", False)),
            "fatal_admission": bool((skills_cfg.get("fatal_guard") or {}).get("admission", False)),
            "judge_pool": int(config.get("system2.judge_pool", 0) or 0) > 0,
            "dedup": bool(skills_cfg.get("dedup", False)),
            "kinds": ["procedure", "state"],
            "world_model": bool((skills_cfg.get("world_model") or {}).get("enabled", False)),
        })

    def work():
        while True:
            job = service.next_job()
            if job is not None:
                service.run_job(job)
            elif service.finished():
                return

    workers = [threading.Thread(target=work, name=f"s2-review-{i}", daemon=True)
               for i in range(max(1, int(proc_cfg.get("num_workers", 4))))]
    for t in workers:
        t.start()
    while True:
        event = events.get()
        if event.get("type") == "shutdown":
            break
        try:
            service.handle(event)
        except Exception:
            logger.exception("System-2 failed to handle a %s event", event.get("type"))
    service.close()
    for t in workers:
        t.join()
    service.files.close()
    results.put({"type": "stopped"})


class S2Process:
    """System-1's handle on the System-2 process (process or thread)."""

    def __init__(self, config, env_name: str, record_dir: Optional[str] = None,
                 mode: str = "process"):
        self.mode = str(mode or "process").lower()
        args = (config.data, env_name, record_dir)
        if self.mode == "thread":
            self._events, self._results = queue.Queue(), queue.Queue()
            self._worker = threading.Thread(
                target=serve, name="system2",
                args=args + (self._events, self._results,
                             logging.getLogger().getEffectiveLevel()), daemon=True)
        else:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            self._events, self._results = ctx.Queue(), ctx.Queue()
            self._worker = ctx.Process(
                target=serve, name="system2",
                args=args + (self._events, self._results,
                             logging.getLogger().getEffectiveLevel()), daemon=True)
        self._worker.start()
        self.stopped = False

    def emit(self, event: Dict[str, Any]) -> None:
        self._events.put(event)

    def poll(self) -> List[Dict[str, Any]]:
        rows = []
        while True:
            try:
                rows.append(self._results.get_nowait())
            except queue.Empty:
                return rows

    def close(self, timeout: float = 1800.0) -> List[Dict[str, Any]]:
        import time

        rows: List[Dict[str, Any]] = []
        if self.stopped:
            return rows
        self._events.put(SHUTDOWN)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                row = self._results.get(timeout=0.5)
            except queue.Empty:
                if not self._worker.is_alive():
                    break
                continue
            if row.get("type") == "stopped":
                break
            rows.append(row)
        self._worker.join(timeout=5.0)
        if self.mode != "thread" and self._worker.is_alive():
            logger.warning("System-2 did not stop within %.0fs; terminating", timeout)
            self._worker.terminate()
            self._worker.join(timeout=5.0)
        if self.mode != "thread":
            for q in (self._events, self._results):
                try:
                    q.cancel_join_thread()
                    q.close()
                except Exception:
                    pass
        self.stopped = True
        return rows
