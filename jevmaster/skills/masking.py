"""Action masking shared by structured-avoid and fatal-action guards.

Masks are applied before System-1 sees the candidate list.  They are a hard
runtime constraint (the model cannot choose the action), not prompt advice:
probability models can be primed by a forbidden action named in a negation.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from jevmaster.skills.retrieval import tokenize
from jevmaster.types import Candidate

# Verbs/modifiers that do not identify the object of an action.  Used by the
# fatal guard to compare action objects without comparing "focus"/"go"/....
GENERIC_ACTION_TOKENS = {
    "focus", "go", "move", "turn", "pick", "take", "put", "place", "drop",
    "open", "close", "toggle", "use", "look", "examine", "craft", "make",
    "connect", "disconnect", "inventory", "item", "object", "the", "a", "an",
}


def pattern_tokens(pattern: str) -> List[str]:
    return tokenize(pattern)


def pattern_matches(action: str, pattern: str, goal_tokens: set[str]) -> bool:
    """Whether ``pattern`` names an action/object in ``action``.

    A pattern that shares a token with the goal is never masked: the goal names
    the legitimate target (e.g. "find the light bulb" must not mask focusing on
    a light bulb).  Multi-token patterns use containment; single-token patterns
    require that token.
    """
    wanted = set(pattern_tokens(pattern))
    if not wanted:
        return False
    if goal_tokens and wanted & goal_tokens:
        return False
    have = set(tokenize(action))
    if len(wanted) == 1:
        return bool(wanted <= have)
    return wanted <= have


def mask_candidates(candidates: Sequence[Candidate], patterns: Iterable[str],
                    goal: str = "") -> Tuple[List[Candidate], List[Tuple[str, str]]]:
    """Drop candidates matched by any pattern, preserving at least one option.

    Returns ``(kept, masked)`` where masked is a list of
    ``(action, pattern)`` pairs.  If every candidate would be removed, the
    original list is returned unchanged (a guard must never make the agent
    unable to act).
    """
    goal_tokens = set(tokenize(goal))
    pats = [str(p).strip() for p in patterns if str(p or "").strip()]
    if not pats or not candidates:
        return list(candidates), []
    kept: List[Candidate] = []
    masked: List[Tuple[str, str]] = []
    for cand in candidates:
        hit = next((p for p in pats if pattern_matches(cand.action, p, goal_tokens)), None)
        if hit is None:
            kept.append(cand)
        else:
            masked.append((cand.action, hit))
    if not kept:
        return list(candidates), []
    return kept, masked
