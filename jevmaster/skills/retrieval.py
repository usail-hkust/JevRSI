"""BM25 retrieval over a skill store (one corpus per kind).

Only retrieval is scored.  Utility is a gate, not a ranking signal: a skill
with three consecutive failed episodes is ``blocked`` and the store's
snapshot excludes it (System-2 can still see it and revise it).
"""
from __future__ import annotations

import difflib
import math
import re
from collections import Counter
from typing import List, Optional, Sequence, Set, Tuple

from jevmaster.skills.skill import Skill
from jevmaster.skills.store import SkillStore
from jevmaster.types import Observation

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "at", "is", "are",
    "it", "this", "that", "with", "for", "you", "your", "i", "be", "as", "by",
    "there", "here", "now", "will", "can", "into", "from", "up", "down", "out",
    "then", "some", "none", "empty",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(str(text or "").lower())
            if len(t) > 1 and t not in _STOPWORDS]


# Scope matching for goal-specific retrieval.  Unlike ``tokenize`` (used by
# BM25, which splits ``m0_i4_10`` into ``m0``/``i4``/``10``), this keeps
# underscore-delimited identifiers intact so two different item ids cannot
# match just because they share a component.  Common task/goal words are
# removed so a stale ``craft-m1-i4-11`` plan cannot match a goal for
# ``m0_i4_10`` through words like "craft" or "items".
_SCOPE_RE = re.compile(r"[a-z0-9_]+")
_SCOPE_STOP = set(_STOPWORDS) | {
    "craft", "following", "items", "item", "goal", "task", "using", "make",
    "create", "put", "place", "pick", "take", "go", "move", "find", "use",
    "look", "examine", "clean", "heat", "cool", "slice", "open", "close",
    "toggle", "finish", "done", "step", "steps", "one", "two", "three",
    "four", "five", "six", "seven", "eight", "nine", "ten",
}


# Counts and quantities ("10", "3x") say nothing about scope: two goals for
# different items both read "2x ...".
_QUANTITY_RE = re.compile(r"\d+[a-z]?")


def scope_tokens(text: str) -> List[str]:
    """Content tokens for goal/skill scope checks (item ids kept whole)."""
    return [t for t in _SCOPE_RE.findall(str(text or "").lower())
            if len(t) > 1 and t not in _SCOPE_STOP and not _QUANTITY_RE.fullmatch(t)]


def goal_identifiers(text: str) -> Set[str]:
    """Opaque identifiers in a goal: tokens mixing letters and digits (``m0_i4_10``).

    Unlike words (colors, objects, substances), an identifier is not a
    parameter a plan can be re-targeted to: its actions name that exact item.
    """
    return {t for t in _SCOPE_RE.findall(str(text or "").lower())
            if len(t) >= 3 and not _QUANTITY_RE.fullmatch(t)
            and re.search(r"[a-z]", t) and re.search(r"\d", t)}


def goal_similarity(source: str, goal: str) -> float:
    """How closely ``goal`` has the form of a plan's ``source`` goal, in [0, 1].

    The larger of two views: content-word overlap (robust to rephrasing such
    as "put a cool apple in X" / "cool some apple and put it in Y") and the
    word-sequence ratio including function words (robust to swapped
    attributes such as "open the blue door" / "open the yellow door").  A
    different goal form ("open the blue door" / "go to the green key") scores
    low on both.  0 when the source names an identifier the goal lacks.
    """
    ids = goal_identifiers(source)
    if ids and not ids <= goal_identifiers(goal):
        return 0.0
    a, b = set(tokenize(source)), set(tokenize(goal))
    overlap = 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0
    wa = _SCOPE_RE.findall(str(source or "").lower())
    wb = _SCOPE_RE.findall(str(goal or "").lower())
    sequence = (difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()
                if wa and wb else 0.0)
    return max(overlap, sequence)


def _goal_form_words(text: str) -> List[str]:
    """Goal words with opaque identifiers/quantities collapsed to ``<id>``."""
    out: List[str] = []
    for tok in _SCOPE_RE.findall(str(text or "").lower()):
        if _QUANTITY_RE.fullmatch(tok):
            continue
        if re.search(r"[a-z]", tok) and re.search(r"\d", tok):
            out.append("<id>")
            continue
        if len(tok) > 1 and tok not in _SCOPE_STOP:
            out.append(tok)
    return out


def goal_form_similarity(source: str, goal: str) -> float:
    """Similarity of goal *form*, ignoring item identifiers and quantities.

    State extractors generalise across items ("craft 2x m0_i4_10" /
    "craft 3x m1_i4_11"); plans must still match identifiers exactly and use
    :func:`goal_similarity` instead.
    """
    wa, wb = _goal_form_words(source), _goal_form_words(goal)
    if not wa and not wb:
        return 1.0 if goal_identifiers(source) and goal_identifiers(goal) else 0.0
    if not wa or not wb:
        return 0.0
    a, b = set(wa), set(wb)
    overlap = 2 * len(a & b) / (len(a) + len(b))
    sequence = difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()
    return max(overlap, sequence)


def goal_head(text: str) -> str:
    """The goal's first content word: the instruction's verb ("put", "go", "open").

    Function words are skipped, verbs are not (``_SCOPE_STOP`` drops them,
    which is right for item scoping but deletes the goal's form).
    """
    for tok in _SCOPE_RE.findall(str(text or "").lower()):
        if len(tok) > 1 and tok not in _STOPWORDS and not _QUANTITY_RE.fullmatch(tok):
            return tok
    return ""


def skill_goal_heads(skill: Skill) -> Set[str]:
    return {h for h in (goal_head(g) for g in (skill.goals or [])) if h}


def single_task_label(skills: Sequence[Skill]) -> bool:
    """True when every scoped skill carries the same task label.

    Then the label cannot tell goal forms apart (MiniGrid: one label for
    go / pick / open / put), and the goal head has to stand in for it.  With
    informative labels (ALFWorld task types, benchmark task labels) the label
    already scopes, and heads are often just phrasings of one task
    ("look at" / "examine", "put a clean X" / "clean some X").
    """
    labels = {t for s in skills for t in (s.tasks or [])}
    return len(labels) == 1


def retrieval_query(observation: Observation, state: str = "") -> str:
    """Query text: task label, goal, inventory, scene, and the current state."""
    return " ".join(p for p in (
        observation.task or "", observation.goal or "", observation.inventory or "",
        observation.text or "", state or "",
    ) if p)


class BM25Retriever:
    def __init__(self, store: SkillStore, top_k: int = 3, k1: float = 1.5,
                 b: float = 0.75, keyword_boost: int = 2):
        self.store = store
        self.top_k = max(1, int(top_k))
        self.k1 = float(k1)
        self.b = max(0.0, min(1.0, float(b)))
        self.keyword_boost = max(1, int(keyword_boost))

    def _doc_tokens(self, skill: Skill) -> List[str]:
        boost = tokenize(" ".join([skill.name, " ".join(skill.keywords), skill.when]))
        return boost * self.keyword_boost + tokenize(skill.retrieval_body())

    def query_scored(self, text: str, kind: str, task: Optional[str] = None,
                     include_blocked: bool = False) -> List[Tuple[float, Skill]]:
        q_terms = tokenize(text)
        skills = self.store.snapshot(kind=kind, task=task, include_blocked=include_blocked)
        if not skills or not q_terms:
            return []
        docs = [self._doc_tokens(s) for s in skills]
        doc_lens = [len(d) for d in docs]
        n = len(docs)
        avgdl = sum(doc_lens) / n if n else 0.0
        term_set = set(q_terms)
        doc_sets = [set(d) for d in docs]
        df = {t: sum(1 for ds in doc_sets if t in ds) for t in term_set}

        scored: List[Tuple[float, Skill]] = []
        for i, tokens in enumerate(docs):
            tf = Counter(tokens)
            dl = doc_lens[i]
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                d_f = df.get(term, 0)
                if not f or not d_f:
                    continue
                idf = math.log(1 + (n - d_f + 0.5) / (d_f + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * (dl / avgdl if avgdl else 1.0))
                score += idf * f * (self.k1 + 1) / denom
            if score > 0.0:
                scored.append((score, skills[i]))
        scored.sort(key=lambda row: (-row[0], row[1].name))
        return scored

    def query(self, text: str, kind: str, task: Optional[str] = None,
              top_k: Optional[int] = None,
              include_blocked: bool = False) -> List[Skill]:
        k = self.top_k if top_k is None else max(1, int(top_k))
        return [s for _score, s in
                self.query_scored(text, kind, task, include_blocked)[:k]]

    def best(self, text: str, kind: str, task: Optional[str] = None,
             min_score: float = 0.0) -> Optional[Skill]:
        rows = self.query_scored(text, kind, task)
        if not rows or rows[0][0] <= min_score:
            return None
        return rows[0][1]


def build_retriever(store: SkillStore, config: dict) -> BM25Retriever:
    return BM25Retriever(
        store,
        top_k=int(config.get("top_k", 3)),
        k1=float(config.get("bm25_k1", 1.5)),
        b=float(config.get("bm25_b", 0.75)),
        keyword_boost=int(config.get("bm25_keyword_boost", 2)),
    )
