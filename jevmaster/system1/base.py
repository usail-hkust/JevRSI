"""System-1 interface.

System-1 always drives execution: given the current observation, the legal
candidates, and the retrieved skills, it returns exactly one action plus a
confidence. It never defers control to System-2.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from jevmaster.skills.skill import Skill
from jevmaster.types import Candidate, Observation, S1Decision


def limit_candidates(candidates: List[Candidate], observation: Observation, limit: int) -> List[Candidate]:
    """Cap the options of one choice question (shared by every Jev-driven mode).

    Above ``limit``, keep the candidates sharing the most tokens with the
    task / goal / inventory / observation text.
    """
    if len(candidates) <= limit:
        return candidates
    from jevmaster.skills.retrieval import retrieval_query, tokenize

    ctx_tokens = set(tokenize(retrieval_query(observation)))
    scored = sorted(
        candidates,
        key=lambda c: len(set(tokenize(c.action)) & ctx_tokens),
        reverse=True,
    )
    return scored[:limit]


class System1(ABC):
    backend: str = "base"

    @abstractmethod
    def select_action(
        self,
        observation: Observation,
        candidates: List[Candidate],
        skills: List[Skill],
        history: Optional[List[str]] = None,
        state_memory: Optional[Dict[str, Any]] = None,
        hint: str = "",
    ) -> S1Decision:
        """Pick one action among ``candidates`` and report a confidence.

        ``history`` is the recent action sequence (agent-side), for backends
        that condition on it; observation-only environments (e.g. ``look
        around`` loops) leave the state unchanged, so without history a
        stateless policy repeats the same action forever. ``state_memory`` is
        the episode's structured state maintained by a state program (None:
        no program for this episode). ``hint`` is an optional instruction
        appended to the choice instructions (an optional decision hint).
        """

    def close(self) -> None:
        return None
