"""System-1 backed by a Jev-style decision model.

Request format follows the Jev-paper conventions (jev-1.13.0, TypeSafe 2026)
and is IDENTICAL for every environment: a structured JSON ``state``
(task / goal / current_observation / inventory / state_memory / history of
command->output pairs / step / relevant_skills -- domain fields, in the
paper's per-domain style; state_memory only when a state program is pinned)
plus one ``choice`` question whose options carry neutral keys
``c0, c1, ...`` with the option's own text as the description (the paper's
ALFWorld condition). No lookahead-derived information is included (no
predicted outputs, no already-seen marks).

Two action-space modes (``system1.jev.two_stage``, default on):

* **Flat** – one choice over the full candidate list (environments without
  template labels, e.g. ALFWorld and the mock env).
* **Two-stage** – when candidates carry ``template`` labels (e.g. ALFWorld),
  the decision is factorized: stage 1 chooses the action *TYPE*
  (``focus on OBJ``, ...), stage 2 chooses the specific action within that
  type's group. Each question uses the same neutral-key format.

``confidence`` is the *weakest stage* (``min(p_template, p_action)``), so the
trigger fires when System-1 is uncertain at either stage.

"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

from jevmaster.skills.skill import Skill
from jevmaster.system1.base import System1
from jevmaster.system1.clients.jev_client import JevClient
from jevmaster.types import Candidate, Observation, S1Decision

# How many command->output history turns ride in the state (the paper sends
# the full history; games here run to 30 steps so we bound the context).
_HISTORY_TURNS = 8


class JevSystem1(System1):
    backend = "jev"

    def __init__(self, client: JevClient, max_candidates: int = 255, two_stage: bool = True):
        self.client = client
        self.max_candidates = max_candidates
        self.two_stage = two_stage

    # ---- request building -------------------------------------------------

    @staticmethod
    def _state_dict(
        observation: Observation,
        skills: List[Skill],
        history: Optional[List[str]],
        state_memory: Optional[dict] = None,
    ) -> dict:
        """Structured state following the Jev-paper conventions.

        ``history`` entries are ``command -> output`` strings (the paper sends
        each earlier command together with the output it produced). Extra
        fields beyond the ALFWorld set (goal / inventory / relevant_skills)
        are domain fields, as in the paper's per-domain state designs.
        """
        state = {
            "task": observation.task,
            "goal": observation.goal,
            "current_observation": observation.text,
            "step": observation.step,
        }
        if observation.inventory:
            state["inventory"] = observation.inventory
        if state_memory:
            state["state_memory"] = state_memory
        if history:
            state["history"] = history[-_HISTORY_TURNS:]
        if skills:
            state["relevant_skills"] = [s.as_prompt_block() for s in skills]
        return state

    def _build_state(
        self,
        observation: Observation,
        skills: List[Skill],
        history: Optional[List[str]] = None,
        state_memory: Optional[dict] = None,
    ) -> str:
        return json.dumps(
            self._state_dict(observation, skills, history, state_memory), ensure_ascii=False
        )

    def _stage_instructions(self, observation: Observation, base: str) -> str:
        # Env cautions ride along in the choice question's dedicated
        # instruction channel -- for the Jev policy this moves candidate
        # probabilities more than state-text alone (A/B tested 2026-10-16).
        if observation.notes:
            return f"{base} {observation.notes}"
        return base

    def _build_instructions(self, observation: Observation, hint: str = "") -> str:
        # The paper's ALFWorld instruction, generalized off "household".
        base = self._stage_instructions(
            observation,
            "You control an agent in a text game. Choose the next command that "
            "makes the most progress toward completing the task. Use history "
            "to remember what you have already seen and done, avoid "
            "revisiting places you already checked, and pick up / use "
            "objects as the task requires.",
        )
        return f"{base} {hint}" if hint else base

    # ---- choice plumbing --------------------------------------------------

    def _choice(
        self,
        state: str,
        options: List[str],
        qid: str,
        instructions: str,
        criteria: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, float, Dict[str, float], dict]:
        """One choice question; returns (option, confidence, probs, raw).

        Options are keyed ``c0, c1, ...`` with the option's own text as the
        description (the paper's ALFWorld convention -- no lookahead-derived
        information); the answer is translated back from key space.
        """
        keys = [f"c{i}" for i in range(len(options))]
        # Each option's description is what the choice question renders next to
        # it; a candidate effect replaces the bare action text when present.
        criteria = {k: (criteria or {}).get(opt, opt) for k, opt in zip(keys, options)}
        value, confidence, probs, raw = self.client.choice(
            state, keys, qid=qid, instructions=instructions, criteria=criteria
        )
        option_of = dict(zip(keys, options))
        value = option_of.get(value, value if value in options else None)
        probs = {option_of.get(k, k): p for k, p in probs.items()}
        return value, float(confidence), probs, raw

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _usage(raw: dict) -> Tuple[int, int]:
        """(tokens_in, tokens_out) from a raw gateway response."""
        usage = raw.get("usage") if isinstance(raw, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        return (
            int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        )

    @staticmethod
    def _legalize(value: str, confidence: float, probs: Dict[str, float], options: List[str]):
        """Snap an answer to a legal option; fallback to argmax, then first."""
        if value in options:
            return value, float(confidence)
        legal_probs = {o: probs.get(o, 0.0) for o in options}
        if any(v > 0 for v in legal_probs.values()):
            value = max(legal_probs, key=legal_probs.get)
            return value, legal_probs[value]
        return options[0], 0.0

    # ---- entry point ------------------------------------------------------

    def select_action(
        self,
        observation: Observation,
        candidates: List[Candidate],
        skills: List[Skill],
        history: Optional[List[str]] = None,
        state_memory: Optional[dict] = None,
        hint: str = "",
    ) -> S1Decision:
        if not candidates:
            return S1Decision(action="", confidence=0.0, backend=self.backend)

        templates: List[str] = []
        seen = set()
        for c in candidates:
            if c.template and c.template not in seen:
                seen.add(c.template)
                templates.append(c.template)

        if self.two_stage and len(templates) > 1:
            return self._select_two_stage(
                observation, candidates, skills, templates, history,
                state_memory, hint,
            )
        return self._select_flat(observation, candidates, skills,
                                 history, state_memory, hint)

    # ---- flat (single choice) ---------------------------------------------

    def _select_flat(
        self,
        observation: Observation,
        candidates: List[Candidate],
        skills: List[Skill],
        history: Optional[List[str]] = None,
        state_memory: Optional[dict] = None,
        hint: str = "",
    ) -> S1Decision:
        options = [c.action for c in candidates[: self.max_candidates]]
        state = self._build_state(observation, skills, history, state_memory)
        instructions = self._build_instructions(observation, hint)

        start = time.perf_counter()
        criteria = ({c.action: c.description for c in candidates[: self.max_candidates]
                     if c.description} or None)
        value, confidence, probs, raw = self._choice(
            state, options, qid="action", instructions=instructions, criteria=criteria,
        )
        latency = (time.perf_counter() - start) * 1000.0

        tokens_in, tokens_out = self._usage(raw)
        value, confidence = self._legalize(value, confidence, probs, options)

        decision = S1Decision(
            action=value,
            confidence=confidence,
            probs=probs,
            latency_ms=latency,
            tokens=tokens_in + tokens_out,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            backend=self.backend,
        )
        return decision

    # ---- two-stage (template -> object) -----------------------------------

    def _select_two_stage(
        self,
        observation: Observation,
        candidates: List[Candidate],
        skills: List[Skill],
        templates: List[str],
        history: Optional[List[str]] = None,
        state_memory: Optional[dict] = None,
        hint: str = "",
    ) -> S1Decision:
        start = time.perf_counter()
        tokens_in = tokens_out = 0
        state = self._build_state(observation, skills, history, state_memory)

        # Stage 1: choose the action TYPE.
        instructions1 = self._stage_instructions(
            observation,
            "You control an agent in a text game. Choose the "
            "TYPE of the next action that makes the most progress toward "
            "completing the task. Use history to remember what you have "
            "already seen and done.",
        )
        if hint:
            instructions1 = f"{instructions1} {hint}"
        tpl, p_tpl, tpl_probs, raw = self._choice(
            state, templates, qid="action_type", instructions=instructions1,
        )
        tin, tout = self._usage(raw)
        tokens_in += tin
        tokens_out += tout
        tpl, p_tpl = self._legalize(tpl, p_tpl, tpl_probs, templates)

        # Stage 2: choose the specific action within the template group.
        # Jev always completes both stages; confidence uses
        # min(p_template, p_action), including every stage-1-uncertain step.
        group = [c.action for c in candidates if c.template == tpl][: self.max_candidates]
        if not group:
            # Template without members (should not happen): fall back to flat.
            return self._select_flat(observation, candidates, skills,
                                     history, state_memory, hint)

        if len(group) == 1:
            # Only one instantiation of this type -- nothing to choose.
            action, p_action, probs = group[0], 1.0, {group[0]: 1.0}
        else:
            instructions2 = self._stage_instructions(
                observation,
                f"You chose the action type '{tpl}'. Now choose the specific "
                "action that makes the most progress toward completing the "
                "task. Use history to remember what you have already seen "
                "and done.",
            )
            if hint:
                instructions2 = f"{instructions2} {hint}"
            described = {c.action: c.description for c in candidates
                         if c.template == tpl and c.description}
            action, p_action, probs, raw = self._choice(
                state, group, qid="action", instructions=instructions2,
                criteria=described or None,
            )
            tin, tout = self._usage(raw)
            tokens_in += tin
            tokens_out += tout
            action, p_action = self._legalize(action, p_action, probs, group)

        latency = (time.perf_counter() - start) * 1000.0
        decision = S1Decision(
            action=action,
            # Weakest-stage confidence: the trigger fires when System-1 is
            # uncertain at EITHER stage of the decision.
            confidence=float(min(p_tpl, p_action)),
            probs=probs,
            latency_ms=latency,
            tokens=tokens_in + tokens_out,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            backend=self.backend,
            two_stage=True,
            template=tpl,
            p_template=float(p_tpl),
            template_probs=tpl_probs,
            p_action=float(p_action),
        )
        return decision

    def close(self) -> None:
        self.client.close()
