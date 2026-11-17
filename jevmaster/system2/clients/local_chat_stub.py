"""A deterministic, dependency-free chat client for offline tests and CI.

Used wherever an OpenAI-compatible chat is needed without a service: the LLM
System-1 (``system1.backend: llm`` + ``system1.llm.backend: stub``) and any
test driving the ACTION/CONFIDENCE reply protocol, mirroring what
``StubSystem1`` / ``StubSystem2`` do for the dual-system mode.

It speaks the same output protocol as the real LLM:

* ``ACTION: <one of the candidate actions>``   -- chosen by token overlap with
  the goal/observation (plus an optional ``prefer`` bonus), like StubSystem1.
* an optional ``CONFIDENCE:`` line (``fixed_confidence``), for tests that
  drive the System-2 uncertainty trigger.
* ``SKILL:`` followed by a ```markdown fenced skill block -- emitted every
  ``skill_every``-th turn so tests deterministically cover the skill-making
  path (legacy ReAct protocol, kept for compatibility).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from jevmaster.skills.retrieval import tokenize
from jevmaster.system2.clients.openai_client import ChatResult


@dataclass
class _Parsed:
    goal: str = ""
    observation: str = ""
    candidates: List[str] = None  # type: ignore[assignment]


class StubChatClient:
    def __init__(
        self,
        prefer: Optional[List[str]] = None,
        skill_every: int = 0,
        skill_name_prefix: str = "stub-skill",
        fixed_reply: Optional[str] = None,
        fixed_confidence: Optional[float] = None,
        **_,
    ):
        self.prefer = [p.lower() for p in (prefer or [])]
        self.skill_every = max(0, int(skill_every))
        self.skill_name_prefix = skill_name_prefix
        self.fixed_reply = fixed_reply  # if set, always returned verbatim (test hook)
        # If set, replies also carry a CONFIDENCE line (the LLM System-1
        # protocol), so tests can drive the System-2 uncertainty trigger.
        self.fixed_confidence = fixed_confidence
        self._turns = 0

    # ---- interface parity with OpenAIChatClient ---------------------------
    def chat_messages(self, messages: List[Dict[str, str]]) -> ChatResult:
        start = time.perf_counter()
        self._turns += 1
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        parsed = _parse_user_message(last_user)
        if self.fixed_reply is not None:
            text = self.fixed_reply
        elif self.skill_every and (self._turns - 1) % self.skill_every == 0:
            text = _skill_reply(self.skill_name_prefix, self._turns, parsed)
        else:
            text = f"ACTION: {_choose_action(parsed, self.prefer)}"
            if self.fixed_confidence is not None:
                text += f"\nCONFIDENCE: {float(self.fixed_confidence):.2f}"
        return ChatResult(
            text=text,
            prompt_tokens=len(tokenize(last_user)),
            completion_tokens=len(tokenize(text)),
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def chat(self, system: str, user: str) -> ChatResult:
        return self.chat_messages(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )

    def close(self) -> None:
        return None


# ---- helpers ---------------------------------------------------------------
def _parse_user_message(text: str) -> _Parsed:
    """Extract goal, observation, and the candidate list from a ReAct user turn."""
    goal = _line_value(text, "Goal:")
    observation = _line_value(text, "Observation:")
    candidates: List[str] = []
    in_actions = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Legal actions"):
            in_actions = True
            continue
        if in_actions:
            if stripped.startswith("- "):
                candidates.append(stripped[2:].strip())
            elif stripped:
                break
    return _Parsed(goal=goal, observation=observation, candidates=candidates or ["wait"])


def _line_value(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _choose_action(parsed: _Parsed, prefer: List[str]) -> str:
    context_tokens = set(tokenize(parsed.goal + " " + parsed.observation))
    best, best_score = None, -1.0
    for cand in parsed.candidates:
        score = float(len(set(tokenize(cand)) & context_tokens))
        score += 1.5 * sum(1 for p in prefer if p in cand.lower())
        if score > best_score:
            best, best_score = cand, score
    return best or "wait"


def _yq(value: str) -> str:
    """Quote a string for safe embedding in YAML frontmatter."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _skill_reply(prefix: str, turn: int, parsed: _Parsed) -> str:
    env_status = (parsed.observation or parsed.goal or "unknown environment state")[:160]
    goal = parsed.goal or "(unknown goal)"
    return (
        "SKILL:\n"
        "```markdown\n"
        "---\n"
        f"name: {prefix}-{turn}\n"
        "env: unknown\n"
        "keywords: [stub, react, skill]\n"
        f"applicability: {_yq(f'situations like pursuing: {goal}')}\n"
        f"environment_status: {_yq(env_status)}\n"
        "---\n"
        "## When to use\n"
        f"When pursuing: {goal} and the environment looks similar to the current state.\n\n"
        "## Procedure\n"
        "1. If the goal is unmet, prefer actions that move the environment toward it.\n"
        "2. Once the goal is satisfied, prefer a closing/finishing action.\n\n"
        "## Notes\n"
        f"Stub-generated on ReAct turn {turn}.\n"
        "```\n"
    )
