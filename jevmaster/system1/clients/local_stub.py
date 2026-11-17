"""A deterministic, dependency-free System-1 for offline smoke tests and CI.

It mimics a Jev ``choice`` by scoring candidates on token overlap with the goal,
observation, and retrieved skills, then softmax-normalizing to produce a
confidence. Low overlap yields low confidence, which naturally exercises the
System-2 skill-consolidation trigger.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional

from jevmaster.skills.skill import Skill
from jevmaster.skills.retrieval import tokenize
from jevmaster.system1.base import System1
from jevmaster.types import Candidate, Observation, S1Decision


class StubSystem1(System1):
    backend = "stub"

    def __init__(
        self,
        fixed_confidence: Optional[float] = None,
        prefer: Optional[List[str]] = None,
        temperature: float = 0.5,
        **_,
    ):
        self.fixed_confidence = fixed_confidence
        self.prefer = [p.lower() for p in (prefer or [])]
        self.temperature = max(temperature, 1e-3)

    def select_action(
        self,
        observation: Observation,
        candidates: List[Candidate],
        skills: List[Skill],
        history=None,
        state_memory=None,
        hint: str = "",
    ) -> S1Decision:
        start = time.perf_counter()
        if not candidates:
            return S1Decision(action="", confidence=0.0, backend=self.backend)

        context_tokens = set(
            tokenize(observation.as_context())
            + tokenize(" ".join(s.text_for_retrieval() for s in skills))
        )
        self.last_state_memory = state_memory   # observable in tests

        scores: List[float] = []
        for cand in candidates:
            cand_tokens = set(tokenize(cand.action))
            overlap = len(cand_tokens & context_tokens)
            bonus = 1.5 if any(p in cand.action.lower() for p in self.prefer) else 0.0
            scores.append(overlap + bonus)

        probs = _softmax(scores, self.temperature)
        best_idx = max(range(len(candidates)), key=lambda i: probs[i])
        action = candidates[best_idx].action
        confidence = self.fixed_confidence if self.fixed_confidence is not None else probs[best_idx]
        prob_map = {candidates[i].action: probs[i] for i in range(len(candidates))}
        latency = (time.perf_counter() - start) * 1000.0
        return S1Decision(
            action=action,
            confidence=float(confidence),
            probs=prob_map,
            latency_ms=latency,
            backend=self.backend,
        )


def _softmax(scores: List[float], temperature: float) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]
