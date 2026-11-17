"""System-1 backed by an LLM (``system1.backend: llm``).

The LLM is a drop-in replacement for the Jev policy: same
:meth:`System1.select_action` contract, same state fields (task, goal, current
observation, inventory, the episode's state memory, recent history, relevant
skills), same candidate list, and a confidence that drives the System-2
uncertainty trigger exactly like Jev's. The only difference is who decides.

The chat client is the shared one (:func:`jevmaster.system2.build_openai_client`,
i.e. the ``system2.llm`` Qwen service); ``system1.llm`` overrides any of its
keys (temperature, thinking, ...).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from jevmaster.skills.skill import Skill
from jevmaster.system1.base import System1
from jevmaster.system1.jev import _HISTORY_TURNS
from jevmaster.types import Candidate, Observation, S1Decision

logger = logging.getLogger(__name__)

LLM_S1_SYSTEM_PROMPT = """You control an agent in a text game, one action per turn. \
Choose the next command that makes the most progress toward completing the \
task. Use history to remember what you have already seen and done, avoid \
revisiting places you already checked, and pick up / use objects as the task \
requires.

Each turn you see the task, the current environment state, the state memory \
maintained for this episode, the most relevant learned skills, your recent \
history, and the exact list of legal actions. The legal actions depend on the \
current environment state and change as it changes -- always choose from the \
list shown this turn.

Reply with EXACTLY two lines:
ACTION: <one action copied EXACTLY from the legal actions list>
CONFIDENCE: <a number between 0.00 and 1.00: how sure you are this is the best \
action now>"""

_MAX_ATTEMPTS = 3
_ACTION_RE = re.compile(r"^\s*ACTION\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"^\s*CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE | re.MULTILINE)

_NO_SKILLS = "Relevant learned skills: (none)"

# The pure LLM baseline (run.mode: llm): no skills or state memory and no
# hints -- only the agent-environment interaction context and the legal
# actions. Kept separate from the aligned llm_agent prompt so a pure baseline
# never sees harness fields it does not use.
PURE_LLM_SYSTEM_PROMPT = """You control an agent in a text environment, one action per turn. \
Use the interaction history to make progress. Choose exactly one of the legal \
actions shown this turn; the legal actions change as the environment changes.

Reply with EXACTLY two lines:
ACTION: <one action copied EXACTLY from the legal actions list>
CONFIDENCE: <a number between 0.00 and 1.00: how sure you are this is the best \
action now>"""


def build_pure_llm_s1_user_message(
    observation: Observation,
    candidates: List[Candidate],
    step: int,
    history: Optional[List[str]] = None,
) -> str:
    """The pure baseline turn: interaction context + legal actions only."""
    parts = [f"[Step {step}]"]
    if observation.task:
        parts.append(f"Task: {observation.task}")
    parts.append(f"Goal: {observation.goal or '(none)'}")
    parts.append(f"Inventory: {observation.inventory or '(empty)'}")
    parts.append(f"Observation: {observation.text}")
    if observation.notes:
        parts.append(f"Note: {observation.notes}")
    turns = (history or [])[-_HISTORY_TURNS:]
    if turns:
        parts.append("Recent history (command -> output):\n" + "\n".join(turns))
    cand_lines = "\n".join(f"- {c.action}" for c in candidates) or "- (none)"
    parts.append("Legal actions (choose exactly one):\n" + cand_lines)
    return "\n\n".join(parts)


def build_llm_s1_user_message(
    observation: Observation,
    candidates: List[Candidate],
    skills: List[Skill],
    step: int,
    history: Optional[List[str]] = None,
    state_memory: Optional[Dict[str, Any]] = None,
    hint: str = "",
) -> str:
    """One decision turn: the same fields Jev's state holds, as labeled text."""
    import json

    parts = [f"[Step {step}]"]
    if observation.task:
        parts.append(f"Task: {observation.task}")
    parts.append(f"Goal: {observation.goal or '(none)'}")
    parts.append(f"Inventory: {observation.inventory or '(empty)'}")
    parts.append(f"Observation: {observation.text}")
    if observation.notes:
        # Environment cautions (e.g. task-ending actions) ride along in the
        # same channel Jev's instructions use.
        parts.append(f"Note: {observation.notes}")
    if state_memory:
        parts.append("State memory (maintained for this episode):\n"
                     + json.dumps(state_memory, ensure_ascii=False))
    if hint:
        parts.append(f"Hint: {hint}")
    if skills:
        blocks = ["Relevant learned skills (matched to the current situation):"]
        blocks.extend(s.as_prompt_block() for s in skills)
        parts.append("\n".join(blocks))
    else:
        parts.append(_NO_SKILLS)
    turns = (history or [])[-_HISTORY_TURNS:]
    if turns:
        parts.append("Recent history (command -> output):\n" + "\n".join(turns))
    cand_lines = "\n".join(f"- {c.action}" for c in candidates) or "- (none)"
    parts.append("Legal actions (choose exactly one):\n" + cand_lines)
    return "\n\n".join(parts)


def parse_llm_s1_reply(text: str, legal: List[str]) -> tuple:
    """(action or None, confidence) from an ACTION / CONFIDENCE reply.

    The action must be one of ``legal``; a confidence line is optional
    (missing: certain, so the System-2 trigger does not fire on format quirks).
    """
    matches = _ACTION_RE.findall(text or "")
    action = None
    for candidate in reversed(matches):           # the last ACTION line counts
        if candidate in legal:
            action = candidate
            break
    conf_matches = _CONFIDENCE_RE.findall(text or "")
    confidence = 1.0
    if conf_matches:
        try:
            confidence = min(1.0, max(0.0, float(conf_matches[-1])))
        except ValueError:
            confidence = 1.0
    return action, confidence


class LLMSystem1(System1):
    """The LLM decides, Jev's contract: one action + a confidence per step."""

    backend = "llm"

    def __init__(self, client, step: int = 0, pure: bool = False):
        self.client = client         # .chat(system, user) -> ChatResult
        self._step = step
        self.pure = bool(pure)       # pure baseline: context + actions only

    def select_action(
        self,
        observation: Observation,
        candidates: List[Candidate],
        skills: Optional[List[Skill]] = None,
        history: Optional[List[str]] = None,
        state_memory: Optional[Dict[str, Any]] = None,
        hint: str = "",
    ) -> S1Decision:
        legal = [c.action for c in candidates]
        if self.pure:
            user = build_pure_llm_s1_user_message(
                observation, candidates, self._step, history)
            system_prompt = PURE_LLM_SYSTEM_PROMPT
        else:
            user = build_llm_s1_user_message(observation, candidates, skills or [], self._step,
                                             history, state_memory, hint)
            system_prompt = LLM_S1_SYSTEM_PROMPT
        action, confidence, reply = None, 1.0, None
        prompt = user
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                reply = self.client.chat(system_prompt, prompt)
            except Exception as exc:  # service failure: report, keep deciding
                logger.warning("LLM System-1 chat failed (attempt %d): %s", attempt, exc)
                break
            action, confidence = parse_llm_s1_reply(reply.text, legal)
            if action is not None:
                break
            legal_sample = "; ".join(legal[:8]) + ("; ..." if len(legal) > 8 else "")
            prompt = (f"Your last reply was not a legal action from the list. Reply with exactly "
                      f"two lines:\nACTION: <one action copied EXACTLY from: {legal_sample}>\n"
                      f"CONFIDENCE: <0.00-1.00>")
        if action is None:
            # Unparsable after the retries: fall back to the LLM's literal last
            # ACTION line (the environment answers illegal actions with
            # feedback) or the first candidate, and keep the episode alive.
            last = _ACTION_RE.findall(reply.text or "") if reply is not None else []
            fallback = last[-1].strip() if last else legal[0]
            logger.warning("LLM System-1 reply unparsable; executing %r", fallback)
            action = fallback
        self._step += 1
        return S1Decision(
            action=action,
            confidence=confidence,
            probs={action: confidence},
            latency_ms=reply.latency_ms if reply else 0.0,
            tokens=(reply.prompt_tokens + reply.completion_tokens) if reply else 0,
            tokens_in=reply.prompt_tokens if reply else 0,
            tokens_out=reply.completion_tokens if reply else 0,
            backend=self.backend,
        )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()
