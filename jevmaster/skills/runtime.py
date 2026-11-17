"""Per-episode wiring of the three skill kinds.

System-1 sees:

* ``procedures``  -- BM25 top-k procedural skills, refreshed every step;
* ``state_memory`` -- a summary of the persisted exploration graph, maintained
  by the pinned state program (facts: rooms, events, regions, ...);
* (M5.3) action effects -- the environment's world model's prediction for each
  legal action, shown next to it.  The prediction for the executed action is checked
  against the next step, and train transitions are kept per task as the
  world model's memory and test set.

A revision published by System-2 reaches later episodes (the state program is
pinned at episode start; procedures are re-retrieved every step).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set

from jevmaster.skills.retrieval import (BM25Retriever, goal_form_similarity,
                                         retrieval_query, scope_tokens)
from jevmaster.skills.skill import Skill, slugify
from jevmaster.skills.state import SandboxError, StateGraph, StateProgram
from jevmaster.skills.store import SkillStore
from jevmaster.skills.world import TransitionBuffer, WorldModel, check, make_row, outcome_view
from jevmaster.types import Observation


def _scope_meta_tokens(skill: Skill, task_tokens: Set[str]) -> Set[str]:
    """Goal/item tokens a skill's own metadata declares (name/when/keywords/expected)."""
    text = " ".join([
        skill.name or "",
        skill.when or "",
        " ".join(skill.keywords or []),
        skill.expected or "",
    ])
    return set(scope_tokens(text)) - task_tokens


class EpisodeMemory:
    def __init__(self, store: SkillStore, retriever: BM25Retriever, task: str,
                 goal: str, config: dict):
        self.store = store
        self.retriever = retriever
        self.task = task
        self.goal = goal
        self.top_k = int(config.get("top_k", 3))
        self.avoid_guard = bool(config.get("avoid_guard", False))
        state_cfg = config.get("state") or {}
        self.budget_ms = float(state_cfg.get("step_budget_ms", 50))
        self.max_lines = int(state_cfg.get("max_lines", 20000))
        self.max_program_chars = int(state_cfg.get("max_program_chars", 6000))
        self.view_chars = int(state_cfg.get("view_chars", 1200))
        self.graph_nodes = int(state_cfg.get("graph_nodes", 12))
        self.state_goal_similarity = float(state_cfg.get("goal_similarity", 0.5))
        self.state_require_goal_match = bool(state_cfg.get("require_goal_match", False))
        self.state_fallback_same_task = bool(state_cfg.get("fallback_same_task", False))
        self.state_surface_facts = bool(state_cfg.get("surface_facts", False))
        self.state_credit_informative = bool(state_cfg.get("credit_informative", False))
        self.state_min_known_ratio = float(state_cfg.get("min_known_ratio", 0.25))
        self.max_facts_chars = int(state_cfg.get("max_facts_chars", 800))

        self.state_skill: Optional[Skill] = None
        self.program: Optional[StateProgram] = None
        self.graph: Optional[StateGraph] = None
        self.state_error = ""
        self._state_used = False
        self.state_data: Any = None
        self._state_labels: List[str] = []

        world_cfg = config.get("world_model") or {}
        self.world_enabled = bool(world_cfg.get("enabled", False))
        self.world_cfg = world_cfg
        self.world_skill: Optional[Skill] = None
        self.world: Optional[WorldModel] = None
        self.world_error = ""
        self.world_errors = 0
        self.world_checks = 0
        self.world_hits = 0
        self._world_rows: List[Dict[str, Any]] = []     # earlier train transitions
        self._episode_rows: List[Dict[str, Any]] = []   # this episode's transitions
        self._predictions: Dict[str, Dict[str, Any]] = {}

        self._shown: Set[str] = set()
        self._procedures: List[Skill] = []
        self._history: List[str] = []

    # ---- setup -----------------------------------------------------------
    def start(self, observation: Observation) -> None:
        self.store.refresh()
        query = retrieval_query(observation)
        self._pin_state(observation, query)
        if self.world_enabled:
            self._pin_world()

    def _pin_state(self, observation: Observation, query: str) -> None:
        skill: Optional[Skill] = None
        if self.state_require_goal_match:
            goal = observation.goal or ""
            if not goal.strip():
                return
            task_tokens = set(scope_tokens(self.task))
            goal_tokens = set(scope_tokens(goal)) - task_tokens
            situation = {s.name: score for score, s in
                         self.retriever.query_scored(query, "state", self.task)}
            best = None
            for candidate in self.store.snapshot(kind="state", task=self.task):
                if candidate.goals:
                    similarity = max(goal_form_similarity(g, goal) for g in candidate.goals)
                    if similarity < self.state_goal_similarity:
                        continue
                elif goal_tokens & _scope_meta_tokens(candidate, task_tokens):
                    similarity = self.state_goal_similarity
                else:
                    continue
                key = (similarity, situation.get(candidate.name, 0.0))
                if best is None or key > best[0] or (key == best[0] and candidate.name < best[1].name):
                    best = (key, candidate)
            if best is not None:
                skill = best[1]
            elif self.state_fallback_same_task:
                skill = self._best_state(query)
        else:
            skill = self._best_state(query)
        if skill is None:
            return
        try:
            program = StateProgram.compile(
                skill.program_code(), self.max_program_chars,
                self.budget_ms, self.max_lines)
        except SandboxError as exc:
            self.state_error = str(exc)
            return
        path = self.store.graph_dir(skill.name) / f"{slugify(self.task)}.json"
        graph = StateGraph.load(path, skill.name, self.task)
        self.state_skill, self.program, self.graph = skill, program, graph
        self._state_used = True
        self._update_state(observation.text, observation.inventory, action=None, step=0)
        self._save_graph()

    def _best_state(self, query: str) -> Optional[Skill]:
        """Highest-BM25 state skill that passes the goal-head scope."""
        for score, skill in self.retriever.query_scored(query, "state", self.task):
            if score > 0.0:
                return skill
        return None



    def _pin_world(self) -> None:
        """The environment's world model (one serves every task; the best live
        accuracy wins if several exist).  Its memory is this task's train
        transitions, or every task's for a task label it has never seen."""
        best = None
        for skill in self.store.snapshot(kind="world_model"):
            key = ((skill.hits + 1.0) / (skill.checks + 2.0), skill.replay_accuracy, skill.name)
            if best is None or key > best[0]:
                best = (key, skill)
        if best is not None:
            skill = best[1]
            try:
                self.world = WorldModel.compile(
                    skill.program_code(),
                    int(self.world_cfg.get("max_program_chars", 20000)),
                    float(self.world_cfg.get("budget_ms", 300)),
                    int(self.world_cfg.get("max_lines", 500000)),
                    int(self.world_cfg.get("effect_chars", 160)))
                self.world_skill = skill
            except SandboxError as exc:
                self.world_error = str(exc)
        capacity = int(self.world_cfg.get("buffer_rows", 400))
        path = self.store.world_path(self.task)
        if path.exists():
            self._world_rows = TransitionBuffer(path, capacity).load()
        else:
            rows: List[Dict[str, Any]] = []
            for task in self.store.world_tasks():
                rows.extend(TransitionBuffer(self.store.world_path(task), capacity).load())
            self._world_rows = rows[-capacity:]

    def world_predictions(self, observation: Observation,
                          actions: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """{action: prediction} for this step; fail-open ({} on an error)."""
        self._predictions = {}
        if self.world is None:
            return {}
        rows = self._world_rows + self._episode_rows
        try:
            self._predictions = self.world.predict(
                self.goal, observation.text, observation.inventory, self._history[-30:],
                list(actions), outcome_view(rows, len(self._episode_rows),
                                            int(self.world_cfg.get("memory", 200))))
        except SandboxError as exc:
            self.world_errors += 1
            self.world_error = str(exc)
        return self._predictions

    def observe_world(self, observation: Observation, actions: Sequence[str], action: str,
                      result: Observation, reward: float, done: bool) -> tuple:
        """Record the transition and check its prediction: (prediction, ok, failed)."""
        if not self.world_enabled:
            return None, None, []
        self._episode_rows.append(make_row(
            self.goal, observation.text, observation.inventory, self._history,
            actions, action, result.text, result.inventory, reward, done))
        pred = self._predictions.get(action)
        ok, failed = check(pred, observation.text, result.text, result.inventory, reward, done)
        if ok is not None:
            self.world_checks += 1
            self.world_hits += int(ok)
        return pred, ok, failed

    def finish_world(self, episode_id: str = "") -> None:
        """Append this episode's transitions to the task's buffer (train only)."""
        if not self.world_enabled or self.store.frozen or not self._episode_rows:
            return
        rows = [dict(r, episode=episode_id) for r in self._episode_rows]
        try:
            TransitionBuffer(self.store.world_path(self.task),
                             int(self.world_cfg.get("buffer_rows", 400))).extend(rows)
        except OSError:
            pass

    def world_name(self) -> Optional[str]:
        return self.world_skill.name if self.world is not None else None

    def world_revision(self) -> Optional[int]:
        return self.world_skill.revision if self.world is not None else None

    # ---- per-step --------------------------------------------------------
    def procedures(self, observation: Observation, state_text: str = "") -> List[Skill]:
        self.store.refresh()
        if self.top_k <= 0:            # procedures disabled (top_k: 0)
            self._procedures = []
            return []
        query = retrieval_query(observation, state_text)
        task_tokens = set(scope_tokens(self.task))
        query_tokens = set(scope_tokens(query)) - task_tokens
        selected: List[Skill] = []
        if query_tokens:
            # BM25 ranks, then a scope gate drops procedures whose own
            # metadata shares no goal/observation token with this step
            # (generic procedures otherwise leak across goals).
            gated = [(score, skill) for score, skill in
                     self.retriever.query_scored(query, "procedure", self.task)
                     if query_tokens & _scope_meta_tokens(skill, task_tokens)]
            selected = [skill for _score, skill in gated[: self.top_k]]
        self._procedures = selected
        self._shown.update(s.name for s in self._procedures)
        return self._procedures

    def avoid_patterns(self) -> List[str]:
        """Structured forbidden-action patterns from the currently shown skills.

        ``avoid`` is never rendered into System-1's prompt; it only feeds the
        runtime mask in the agent loop.
        """
        if not self.avoid_guard:
            return []
        out: List[str] = []
        for skill in list(self._procedures):
            out.extend(skill.avoid or [])
        seen = set()
        return [p for p in out if p and not (p in seen or seen.add(p))]





    def state_summary(self) -> str:
        if self.graph is None:
            return ""
        return self.graph.summary(self.graph_nodes, self.view_chars)

    def context(self) -> Dict[str, Any]:
        """The ``state_memory`` payload shown to System-1."""
        out: Dict[str, Any] = {}
        summary = self.state_summary()
        if summary:
            out["state"] = summary
        if (self.state_surface_facts and self.state_data is not None
                and isinstance(self.state_data, dict) and self.state_informative()):
            text = json.dumps(self.state_data, ensure_ascii=False, sort_keys=True)
            out["state_facts"] = (self.state_data if len(text) <= self.max_facts_chars
                                  else text[: max(0, self.max_facts_chars - 3)] + "...")
        return out

    def after_step(self, action: str, observation: Observation, feedback: str,
                   step: int) -> None:
        if self.program is not None:
            self._update_state(observation.text, observation.inventory,
                               action=action, step=step)
            self._save_graph()

    def _update_state(self, observation_text: str, inventory: str,
                      action: Optional[str], step: int) -> None:
        if self.program is None or self.graph is None:
            return
        try:
            label, data = self.program(
                self.goal, observation_text, inventory, self._history)
        except SandboxError as exc:
            self.state_error = str(exc)
            return
        self.state_data = data
        self._state_labels.append(str(label))
        self.graph.observe(label, action=action, step=step)

    def _save_graph(self) -> None:
        if self.graph is not None and self.state_skill is not None:
            path = self.store.graph_dir(self.state_skill.name) / f"{slugify(self.task)}.json"
            try:
                self.graph.save(path)
            except OSError:
                pass

    # ---- credit ----------------------------------------------------------
    def state_informative(self) -> bool:
        if not self._state_labels:
            return False
        unknown = {"", "(unknown)", "unknown", "none", "null"}
        known = sum(1 for label in self._state_labels if label.lower() not in unknown)
        return (known / len(self._state_labels)) >= max(0.0, min(1.0, self.state_min_known_ratio))

    def used_names(self) -> List[str]:
        names = set(self._shown)
        if self._state_used and self.state_skill is not None and self.program is not None:
            if not self.state_credit_informative or self.state_informative():
                names.add(self.state_skill.name)
        return sorted(names)

    def state_name(self) -> Optional[str]:
        return self.state_skill.name if self.program is not None else None





    def set_history(self, history: List[str]) -> None:
        """Recent ``command -> output`` turns, passed to the state program."""
        self._history = history
