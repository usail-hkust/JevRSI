"""Learned fatal-action rules.

Some environments end the episode immediately with a large negative score when
S1 touches the wrong object (focus-graded goals).  The rule table
learns the *action object* that caused that outcome, per task, and masks the
same object in later steps.  It is framework-owned, not System-2 prose, and it
generalises across attributes (yellow/green light bulb) while never masking an
object named by the current goal.

The table lives in ``<skills.dir>/<env>/_fatal_rules.json`` so the normal
train-store snapshot carries it into eval; it is read-only once frozen.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from jevmaster.skills.masking import GENERIC_ACTION_TOKENS
from jevmaster.skills.retrieval import tokenize
from jevmaster.types import Candidate

_FILE = "_fatal_rules.json"


def action_verb(action: str, template: str = "") -> str:
    """The leading action word (``focus``, ``connect``, ``dunk``, ...)."""
    toks = tokenize(action)
    return toks[0] if toks else ""


def action_object_tokens(action: str, template: str = "", task: str = "",
                         goal: str = "") -> List[str]:
    """Tokens that name the object of an action, not its verb/instance/placeholder."""
    remove = set(GENERIC_ACTION_TOKENS)
    remove |= set(tokenize(task))
    remove |= set(tokenize(goal))
    remove |= set(tokenize(template))
    return [t for t in tokenize(action) if t not in remove]


@dataclass
class FatalRuleTable:
    path: str | Path
    enabled: bool = True
    threshold: float = -99.0
    min_support: int = 1
    min_remaining: int = 1
    require_object_match: bool = True
    frozen: bool = False
    max_examples: int = 3
    _rules: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load()

    # ---- persistence -----------------------------------------------------
    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            rules = raw.get("rules", []) if isinstance(raw, dict) else []
        except (OSError, ValueError):
            rules = []
        self._rules = [r for r in rules if isinstance(r, dict) and r.get("tokens")]

    def _save(self) -> None:
        if self.frozen:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "rules": self._rules}
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.path)

    def _key(self, task: str, verb: str,
             tokens: Iterable[str]) -> Tuple[str, str, Tuple[str, ...]]:
        return str(task or ""), str(verb or ""), tuple(sorted(set(tokens)))

    # ---- learning / matching --------------------------------------------
    def observe(self, task: str, action: str, template: str, goal: str,
                score: float = -100.0) -> Optional[Dict[str, Any]]:
        """Record a fatal action.  No-op when frozen or when it names no object."""
        if not self.enabled or self.frozen or float(score) > float(self.threshold):
            return None
        tokens = action_object_tokens(action, template, task, goal)
        if not tokens:
            return None
        verb = action_verb(action, template)
        key = self._key(task, verb, tokens)
        rule = next((r for r in self._rules
                     if self._key(r.get("task", ""), r.get("verb", ""), r["tokens"]) == key),
                    None)
        if rule is None:
            rule = {"task": str(task or ""), "verb": verb,
                    "tokens": sorted(set(tokens)),
                    "template": str(template or ""), "support": 0, "examples": []}
            self._rules.append(rule)
        rule["support"] = int(rule.get("support", 0)) + 1
        example = " ".join(str(action or "").split())[:160]
        examples = list(rule.get("examples") or [])
        if example and example not in examples:
            examples.append(example)
        rule["examples"] = examples[-self.max_examples:]
        self._save()
        return rule

    def matching_rule(self, task: str, action: str, goal: str,
                      template: str = "") -> Optional[Dict[str, Any]]:
        """The strongest learned rule that should mask this action, if any."""
        if not self.enabled or not self._rules:
            return None
        cand_tokens = set(action_object_tokens(action, template, task, goal))
        goal_tokens = set(tokenize(goal))
        if not cand_tokens:
            return None
        best = None
        cand_verb = action_verb(action, template)
        for rule in self._rules:
            if str(rule.get("task", "")) != str(task or ""):
                continue
            if int(rule.get("support", 0)) < int(self.min_support):
                continue
            if rule.get("verb") and cand_verb and str(rule["verb"]) != cand_verb:
                continue
            r_tokens = set(rule.get("tokens") or [])
            if not r_tokens or (goal_tokens & r_tokens):
                continue
            overlap = len(cand_tokens & r_tokens)
            need = min(2, len(r_tokens))
            if self.require_object_match and overlap < need:
                continue
            if not self.require_object_match and not overlap:
                continue
            key = (overlap, int(rule.get("support", 0)))
            if best is None or key > best[0]:
                best = (key, rule)
        return best[1] if best else None

    def mask(self, candidates: Sequence[Candidate], task: str, goal: str,
             templates: Optional[Dict[str, str]] = None) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
        """Drop candidates matching a learned fatal rule, keeping one option."""
        if not self.enabled or not self._rules:
            return list(candidates), []
        templates = templates or {}
        kept: List[Candidate] = []
        masked: List[Dict[str, Any]] = []
        for cand in candidates:
            rule = self.matching_rule(task, cand.action, goal, templates.get(cand.action, ""))
            if rule is None:
                kept.append(cand)
            else:
                masked.append({"action": cand.action, "tokens": list(rule.get("tokens") or [])})
        if len(kept) < max(1, int(self.min_remaining)):
            return list(candidates), []
        return kept, masked
