"""The dual-system agent loop (M5).

System-1 decides every step.  System-2 runs in its own process and is asked
to review low-confidence steps asynchronously; it may write or revise one of
the three skill kinds.  System-1 reads the skill store before every step, so a
published skill is picked up without ever waiting for System-2.

Per step System-1 sees:

* relevant ``procedures`` (BM25 top-k, advice text);
* ``state_memory``: the persisted exploration graph maintained by the pinned
  state extractor (rooms / events / regions);

Utility is per revision: three consecutive failed episodes block a skill from
retrieval; System-2 can revise it to reset the counters.
"""
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

from jevmaster.config import Config
from jevmaster.envs import make_env
from jevmaster.record.experiment import build_recorder
from jevmaster.skills.fatal_guard import FatalRuleTable
from jevmaster.skills.masking import mask_candidates
from jevmaster.skills.retrieval import build_retriever
from jevmaster.skills.runtime import EpisodeMemory
from jevmaster.skills.store import SkillStore
from jevmaster.system1 import build_system1
from jevmaster.system1.base import limit_candidates
from jevmaster.system2 import S2Process
from jevmaster.types import Observation, S1Decision, StepRecord

logger = logging.getLogger(__name__)


class DualSystemAgent:
    def __init__(self, config: Config, *, env=None, system1=None):
        """The dual-system agent: System-1 acts, System-2 maintains the library."""
        self.config = config
        env_cfg = config.section("env")
        self.env_name = env_cfg.get("name", "mock")
        self.step_limit = int(env_cfg.get("step_limit", 100))
        self.candidate_limit = int(config.get("run.candidate_limit", 255))
        # M5.3 world model: a System-2-maintained predictor supplies the effects.
        self.world_model = bool(((config.section("skills").get("world_model") or {})
                                 .get("enabled", False)))
        self.env = env if env is not None else make_env(
            self.env_name, step_limit=self.step_limit, **(env_cfg.get("params") or {}))

        self.skills_cfg = config.section("skills")
        self.lifecycle_cfg = self.skills_cfg.get("lifecycle") or {}
        self.lifecycle_mode = str(self.lifecycle_cfg.get("mode", "absolute") or "absolute").lower()
        self.ablation_rate = max(0.0, min(1.0, float(self.lifecycle_cfg.get("ablation_rate", 0.0))))
        self.store = SkillStore(
            self.skills_cfg.get("dir", "skills_store"), self.env_name,
            frozen=bool(self.skills_cfg.get("freeze")),
            block_after=int(self.skills_cfg.get("block_after", 3)),
            lifecycle=self.lifecycle_cfg)
        seed = f"{config.get('run.seed', 0)}:{self.env_name}:{(config.get('run.mode') or 'dual').lower()}"
        self._rng = random.Random(seed)
        self.fatal_guard = self._build_fatal_guard(self.store)
        self.retriever = build_retriever(self.store, self.skills_cfg)

        self.system1 = system1 if system1 is not None else build_system1(config)
        self.mode_label = (config.get("run.mode") or "dual").lower()
        self.recorder = build_recorder(config, self.mode_label, self.env_name)
        self.uncertain_threshold = float(config.get("trigger.confidence_threshold", 0.5))
        self.run_phase = str(config.get("run.phase", "") or "").lower()

        s2_cfg = config.section("system2")
        self.s2_backend = str(s2_cfg.get("backend") or "stub").lower()
        self.memory_enabled = self.s2_backend != "none"
        self.s2: Optional[S2Process] = None
        if self.memory_enabled:
            self.s2 = S2Process(
                config, self.env_name, record_dir=str(self.recorder.dir),
                mode=(s2_cfg.get("process", {}) or {}).get("mode", "process"))

        self.success_threshold = float(config.get("run.success_threshold", 1e-9))
        self.s2_totals = {
            "requests": 0, "reviews": 0, "declined": 0, "published": 0,
            "rejected": 0, "empty": 0, "frozen": 0, "errors": 0,
            "dropped": 0, "blocked": 0,
        }

    def _build_fatal_guard(self, store: SkillStore) -> Optional[FatalRuleTable]:
        fg = self.skills_cfg.get("fatal_guard") or {}
        if not bool(fg.get("enabled", False)):
            return None
        return FatalRuleTable(
            store.dir / "_fatal_rules.json",
            enabled=True,
            threshold=float(fg.get("threshold", -99.0)),
            min_support=int(fg.get("min_support", 1)),
            min_remaining=int(fg.get("min_remaining", 1)),
            require_object_match=bool(fg.get("require_object_match", True)),
            frozen=bool(store.frozen),
        )

    # ---- System-2 results ------------------------------------------------
    def _collect(self, rows: List[Dict], counts: Optional[Dict[str, int]] = None) -> None:
        for row in rows:
            keys: List[str] = []
            if row.get("type") == "review":
                keys.append("reviews")
                status = row.get("status")
                if status == "error":
                    keys.append("errors")
                elif row.get("decision") == "NONE":
                    keys.append("declined")
                elif status == "published":
                    keys.append("published")
                elif status in self.s2_totals:
                    keys.append(status)          # rejected | empty | frozen
                elif status == "rejected":
                    keys.append("rejected")
            elif row.get("type") == "dropped":
                keys.append("dropped")
            elif row.get("type") == "blocked":
                self.s2_totals["blocked"] += len(row.get("skills") or [])
                if counts is not None:
                    counts["blocked"] = counts.get("blocked", 0) + len(row.get("skills") or [])
            for key in keys:
                self.s2_totals[key] = self.s2_totals.get(key, 0) + 1
                if counts is not None:
                    counts[key] = counts.get(key, 0) + 1
            self.recorder.log_s2_result(row)

    # ---- episode ---------------------------------------------------------
    def run_episode(self, task: str, variation: int, episode_id: str,
                    ) -> Dict:
        """One episode."""
        obs: Observation = self.env.reset(task, variation)
        task_label = obs.task or task
        item_meta = {
            key: obs.extra[key]
            for key in ("item_id", "split", "level", "quest_length", "world_size",
                        "task_type", "data_idx")
            if obs.extra and key in obs.extra
        }
        history: List[str] = []
        counts: Dict[str, int] = {}
        skills_before = len(self.store)
        memory = EpisodeMemory(self.store, self.retriever, task_label,
                               obs.goal, self.skills_cfg)
        # Relative lifecycle uses occasional no-skill episodes to estimate the
        # task baseline.  Never ablate eval (frozen) or non-dual modes.
        self.store.refresh()
        ablate = bool(
            self.lifecycle_mode == "relative"
            and self.ablation_rate > 0.0
            and self.mode_label == "dual"
            and not self.store.frozen
            and self.run_phase in ("train", "")
            and self.store.count() > 0
            and self._rng.random() < self.ablation_rate
        )
        use_memory = self.memory_enabled and not ablate
        if use_memory:
            memory.start(obs)
        if self.s2 is not None:
            self.s2.emit({
                "type": "episode_start", "episode_id": episode_id,
                "task": task_label, "goal": obs.goal,
                "env_task": task, "variation": variation, "item": item_meta,
                "observation": obs.text, "inventory": obs.inventory,
                "state_program": memory.state_name(),
                "ablation": ablate,
                "world_model": memory.world_name(),
                "time": time.time(),
            })

        step = 0
        uncertain_steps = 0
        latency_ms = 0.0
        cumulative_reward = 0.0
        self.last_actions: List[str] = []

        while step < self.step_limit:
            candidates = limit_candidates(self.env.candidates(), obs, self.candidate_limit)
            if not candidates:
                break
            memory.set_history(history)
            effects: Dict[str, str] = {}
            if self.world_model:
                if use_memory:
                    predictions = memory.world_predictions(obs, [c.action for c in candidates])
                    effects = {a: p["effect"] for a, p in predictions.items()}
                    for c in candidates:
                        effect = effects.get(c.action)
                        c.description = f"{c.action} (expected: {effect})" if effect else None
            state_summary = memory.state_summary()
            # Procedures are retrieved before masking, so their structured
            # avoid patterns can remove dangerous candidates from S1 entirely.
            procedures = memory.procedures(obs, state_summary) if use_memory else []
            avoid_patterns = memory.avoid_patterns() if use_memory else []
            candidates, avoid_masked = mask_candidates(candidates, avoid_patterns, obs.goal)
            fatal_masked = []
            if self.fatal_guard is not None and not ablate and self.mode_label == "dual":
                templates = {c.action: getattr(c, "template", "") or "" for c in candidates}
                candidates, fatal_masked = self.fatal_guard.mask(
                    candidates, task_label, obs.goal, templates)
            legal = [c.action for c in candidates]
            state_view = memory.context() if use_memory else None
            hint = ""

            self.store.refresh()
            skill_version = self.store.version
            s1 = self.system1.select_action(
                obs, candidates, procedures, history=history,
                state_memory=state_view, hint=hint)
            executed = s1.action
            if not executed:
                break

            result = self.env.step(executed)
            feedback = result.observation.text
            prediction, prediction_ok, prediction_failed = (
                memory.observe_world(obs, legal, executed, result.observation,
                                     result.reward, result.done)
                if use_memory else (None, None, []))
            fatal_learned = None
            if (self.fatal_guard is not None and getattr(result, "fatal", False)
                    and not ablate and self.run_phase in ("train", "")
                    and not self.store.frozen):
                cand = next((c for c in candidates if c.action == executed), None)
                fatal_learned = self.fatal_guard.observe(
                    task_label, executed, getattr(cand, "template", "") or "",
                    obs.goal, score=float(self.env.score()))
            cumulative_reward += result.reward
            latency_ms += s1.latency_ms
            history.append(_turn(executed, feedback))
            if use_memory:
                memory.after_step(executed, result.observation, feedback, step)
            self.last_actions.append(executed)

            uncertain = (self.s2 is not None
                         and s1.confidence < self.uncertain_threshold)
            uncertain_steps += int(uncertain)
            if self.s2 is not None:
                event = {
                    "type": "step", "episode_id": episode_id, "step": step,
                    "task": task_label, "goal": obs.goal, "action": executed,
                    "confidence": round(float(s1.confidence), 4),
                    "feedback": feedback,
                    "next_inventory": result.observation.inventory,
                    "reward": result.reward,
                    "skills": [p.name for p in procedures],
                    "state": state_summary,
                    "observation": obs.text, "inventory": obs.inventory,
                    "candidates": legal, "uncertain": uncertain,
                    "state_program": memory.state_name(),
                    "effect": effects.get(executed),
                    "done": bool(result.done),
                    "world_model": memory.world_name(),
                    "world_revision": memory.world_revision(),
                    "prediction": prediction,
                    "prediction_ok": prediction_ok,
                    "prediction_failed": prediction_failed,
                }
                self.s2.emit(event)
                self._collect(self.s2.poll(), counts)

            self.recorder.log_step(StepRecord(
                episode_id=episode_id, env_name=self.env_name, task=task_label,
                step=step, goal=obs.goal, observation=obs.text,
                inventory=obs.inventory, candidates=legal,
                s1_action=executed, s1_confidence=s1.confidence, s1_probs=s1.probs,
                s1_latency_ms=s1.latency_ms, s1_tokens=s1.tokens,
                s2_triggered=uncertain, s2_job_id=None, reward=result.reward,
                cumulative_reward=cumulative_reward, done=result.done,
                retrieved_skills=[p.name for p in procedures],
                skill_version=skill_version, s1_backend=s1.backend,
                mode=self.mode_label, exp_id=self.recorder.exp_id,
                next_observation=feedback,
                extra={
                    "s1_tokens_in": s1.tokens_in, "s1_tokens_out": s1.tokens_out,
                    "state_memory": state_view,
                    "state_program": memory.state_name(),
                    "avoid_masked": [a for a, _ in avoid_masked],
                    "avoid_patterns": avoid_patterns,
                    "fatal_masked": fatal_masked,
                    "fatal_learned": fatal_learned,
                    "state_error": memory.state_error or None,
                    "effects_shown": len(effects),
                    "effect_of_action": effects.get(executed),
                    "world_model": memory.world_name(),
                    "prediction": prediction,
                    "prediction_ok": prediction_ok,
                    "prediction_failed": prediction_failed,
                }))

            obs = result.observation
            step += 1
            if result.done:
                break

        score = self.env.score()
        score_max = float(getattr(self.env, "score_max", 1.0) or 1.0)
        progress = max(-1.0, min(1.0, float(score) / score_max if score_max else float(score)))
        if float(score) <= -99.0:
            progress = -1.0
        success = score > self.success_threshold or cumulative_reward > self.success_threshold
        used_skills = memory.used_names() if use_memory else []
        if use_memory:
            memory.finish_world(episode_id)
        if self.s2 is not None:
            self.s2.emit({
                "type": "episode_end", "episode_id": episode_id,
                "success": success, "score": score, "progress": progress,
                "ablation": ablate, "score_max": score_max,
                "used_skills": used_skills,
                "state_program": memory.state_name(),
            })
            self._collect(self.s2.poll(), counts)
        self.store.refresh()
        self.s2_totals["requests"] += uncertain_steps

        summary = {
            "mode": self.mode_label,
            "episode_id": episode_id,
            "exp_id": self.recorder.exp_id,
            "task": task_label,
            "env_task": task,
            "goal": obs.goal,
            "variation": variation,
            "item": item_meta,
            "steps": step,
            "score": score,
            "cumulative_reward": cumulative_reward,
            "success": success,
            "progress": progress,
            "ablation": ablate,
            "s2_requests": uncertain_steps,
            "s2_reviews": counts.get("reviews", 0),
            "s2_triggers": counts.get("published", 0),
            "s2_declined": counts.get("declined", 0),
            "s2_published": counts.get("published", 0),
            "s2_rejected": counts.get("rejected", 0),
            "s2_errors": counts.get("errors", 0),
            "s2_dropped": counts.get("dropped", 0),
            "skills_blocked": counts.get("blocked", 0),
            "s2_totals": dict(self.s2_totals),
            "s1_latency_ms": latency_ms / step if step else 0.0,
            "state_program": memory.state_name(),
            "state_informative": memory.state_informative() if use_memory else False,
            "used_skills": used_skills,
            "fatal": float(score) <= -99.0,
            "state_error": memory.state_error,
            "world_model": memory.world_name(),
            "world_checks": memory.world_checks,
            "world_hits": memory.world_hits,
            "world_errors": memory.world_errors,
            "world_error": memory.world_error or None,
            "skills_added": len(self.store) - skills_before,
            "skills_total": len(self.store),
            "procedures_total": self.store.count("procedure"),
            "states_total": self.store.count("state"),
            "world_models_total": self.store.count("world_model"),
            "blocked_total": self.store.blocked_count(),
            "validated_total": sum(1 for s in self.store.snapshot(include_blocked=True)
                                   if s.validated),
        }
        self.recorder.log_episode(summary)
        logger.info("Episode done: %s", summary)
        return summary

    def run(self) -> List[Dict]:
        pairs = self.env.task_variations()
        episodes = int(self.config.get("run.episodes", len(pairs)) or len(pairs))
        pairs = pairs[:episodes]
        done = self.recorder.completed_episode_ids()
        if done:
            logger.info("resume: %d episode(s) already complete, skipping", len(done))
        summaries = self.recorder.episode_summaries()
        for idx, (task, variation) in enumerate(pairs):
            episode_id = f"{self.env_name}-{idx}-{_safe(task)}-v{variation}"
            if episode_id in done:
                continue
            summaries.append(self.run_episode(task, variation, episode_id))
        return summaries

    def close(self) -> None:
        try:
            if self.s2 is not None:
                self._collect(self.s2.close())
                self.store.refresh()
        finally:
            self.system1.close()
            self.env.close()
            self.recorder.close(final={"s2_totals": dict(self.s2_totals),
                                       "skills_total": len(self.store),
                                       "blocked_total": self.store.blocked_count()})


def _same_text(a: str, b: str) -> bool:
    return " ".join(str(a or "").split()) == " ".join(str(b or "").split())


def _turn(action: str, feedback: str) -> str:
    return f"{action} -> {' '.join(str(feedback).split())[:300]}"


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (text or "task"))[:32]


def build_agent(config: Config):
    """Construct the agent selected by ``run.mode``.

    ``dual`` (Jev + System-2), ``jev`` (System-2 off: no skills), ``llm_agent``
    (the LLM decides; everything else unchanged).

    Thinking policy (defaults):
      * dual: skill development thinks (1024-token cap); the judge stays direct.
        This is the S0c reference configuration;
      * llm_agent: the S1 decision thinks (1024-token cap); judge off,
        develop on;
      * llm: the pure baseline's S1 decision thinks (1024-token cap), no S2.

    ``run.thinking_policy: config`` skips the automatic policy, so the
    explicit ``extra_body.chat_template_kwargs.enable_thinking`` settings win.
    """
    mode = (config.get("run.mode") or "dual").lower()
    if mode == "jev":
        config.set("system2.backend", "none")
    elif mode == "llm":
        # Pure LLM baseline: no skills, no state memory.
        config.set("system1.backend", "llm")
        config.set("system2.backend", "none")
        if (config.get("run.thinking_policy") or "auto").lower() != "config":
            _set_thinking(config, "system1.llm", True)
            config.set("system1.llm.thinking_token_budget", 1024)
            config.set("system1.llm.timeout", 600)
    elif mode == "llm_agent":
        config.set("system1.backend", "llm")
        if (config.get("run.thinking_policy") or "auto").lower() != "config":
            _set_thinking(config, "system1.llm", True)
            config.set("system1.llm.thinking_token_budget", 1024)
            config.set("system1.llm.timeout", 600)
            _set_thinking(config, "system2.llm", False)
            if config.get("system2.llm.develop"):
                _set_thinking(config, "system2.llm.develop", True)
    else:
        # dual: skill development reasons before writing; the judge stays direct.
        # This matches the S0c reference run.
        if (config.get("run.thinking_policy") or "auto").lower() != "config":
            _set_thinking(config, "system2.llm", False)
            config.set("system2.llm.thinking_token_budget", 1024)
            if config.get("system2.llm.develop"):
                _set_thinking(config, "system2.llm.develop", True)
                config.set("system2.llm.develop.thinking_token_budget", 1024)
    if mode not in ("dual", "jev", "llm", "llm_agent"):
        raise ValueError(f"Unknown run.mode: {mode}")
    return DualSystemAgent(config)


def _set_thinking(config: Config, section: str, enabled: bool) -> None:
    """Toggle Qwen/vLLM thinking for one LLM client section."""
    config.set(f"{section}.extra_body.chat_template_kwargs.enable_thinking",
               bool(enabled))
