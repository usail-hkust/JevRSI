"""World model: a System-2-written one-step model of the legal actions.

The program is one function::

    predict(goal, observation, inventory, history, actions, outcomes)
        -> {action: {"effect": str, <checkable fields>}}

``effect`` is the text System-1 sees next to the action.  The checkable
fields are what the framework verifies against the next step:

* ``same``  -- the next observation is identical to the current one;
* ``ends``  -- the episode ends right after the action;
* ``reward`` -- the sign of the step reward: "+", "0" or "-";
* ``see``   -- phrases that will appear in the next observation;
* ``carry`` -- phrases that will appear in the next inventory.

A prediction without any checkable field cannot be verified and is dropped.
One world model serves every task of an environment.  ``outcomes`` is the
framework's memory of executed actions in this task
(earlier train episodes and this episode, most recent first), so the program
can generalize from what really happened.  The same check scores a program
live (the executed action of every step) and at admission (recorded
transitions), and both are defined here.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from jevmaster.skills.state import SandboxError, StateProgram

FIELDS = ("same", "ends", "reward", "see", "carry")
_MAX_PHRASES = 4
_PHRASE_CHARS = 80
_ROW_CHARS = 1500            # observation text kept per stored transition
_VIEW_CHARS = 300            # text per outcome row handed to predict()
_HISTORY_TURNS = 8


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def _sign(value: float) -> str:
    return "+" if value > 1e-9 else ("-" if value < -1e-9 else "0")


def _phrases(value: Any) -> Optional[List[str]]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return None
    out = [_norm(v)[:_PHRASE_CHARS] for v in value if isinstance(v, str) and _norm(v)]
    return out[:_MAX_PHRASES] or None


def normalize(raw: Any, legal: Sequence[str], max_chars: int = 160) -> Dict[str, Dict[str, Any]]:
    """Keep legal actions with an effect text and at least one checkable field."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SandboxError("predict must return a dict {action: {effect, ...}}")
    allowed = set(legal)
    out: Dict[str, Dict[str, Any]] = {}
    for action, value in raw.items():
        if action not in allowed or not isinstance(value, dict):
            continue
        effect = _norm(value.get("effect"))[:max_chars]
        if not effect:
            continue
        pred: Dict[str, Any] = {"effect": effect}
        for key in ("same", "ends"):
            if isinstance(value.get(key), bool):
                pred[key] = value[key]
        reward = value.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            reward = _sign(float(reward))
        if reward in ("+", "0", "-"):
            pred["reward"] = reward
        for key in ("see", "carry"):
            phrases = _phrases(value.get(key))
            if phrases:
                pred[key] = phrases
        if len(pred) > 1:
            out[action] = pred
    return out


def check(pred: Optional[Dict[str, Any]], before: str, after: str, after_inventory: str,
          reward: float, done: bool) -> Tuple[Optional[bool], List[str]]:
    """(correct, failed fields); (None, []) when nothing was predicted."""
    if not pred or not any(k in pred for k in FIELDS):
        return None, []
    failed: List[str] = []
    if "same" in pred and pred["same"] != (_norm(before) == _norm(after)):
        failed.append("same")
    if "ends" in pred and pred["ends"] != bool(done):
        failed.append("ends")
    if "reward" in pred and pred["reward"] != _sign(float(reward or 0.0)):
        failed.append("reward")
    low_after, low_inv = _norm(after).lower(), _norm(after_inventory).lower()
    if "see" in pred and not all(p.lower() in low_after for p in pred["see"]):
        failed.append("see")
    if "carry" in pred and not all(p.lower() in low_inv for p in pred["carry"]):
        failed.append("carry")
    return not failed, failed


def actual(row: Dict[str, Any]) -> Dict[str, Any]:
    """What really happened in a transition row, in the prediction's terms."""
    return {"same": _norm(row.get("observation")) == _norm(row.get("next_observation")),
            "ends": bool(row.get("done")),
            "reward": _sign(float(row.get("reward") or 0.0)),
            "observation": _norm(row.get("next_observation"))[:_VIEW_CHARS],
            "inventory": _norm(row.get("next_inventory"))[:150]}


class WorldModel:
    """A compiled ``predict(goal, observation, inventory, history, actions, outcomes)``."""

    def __init__(self, program: StateProgram, max_chars: int = 160):
        self.program = program
        self.max_chars = int(max_chars)

    @classmethod
    def compile(cls, code: str, max_chars: int = 20000, budget_ms: float = 300.0,
                max_lines: int = 500000, effect_chars: int = 160) -> "WorldModel":
        try:
            program = StateProgram.compile(code, max_chars, budget_ms, max_lines, entry="predict")
        except SandboxError as exc:
            raise SandboxError(str(exc).replace(
                "predict(goal, observation, inventory, history)",
                "predict(goal, observation, inventory, history, actions, outcomes)")) from exc
        return cls(program, effect_chars)

    def predict(self, goal: str, observation: str, inventory: str, history: Sequence[str],
                actions: Sequence[str], outcomes: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        raw = self.program._traced(self.program.fn, goal or "", observation or "",
                                   inventory or "", list(history or []), list(actions),
                                   [dict(o) for o in outcomes])
        return normalize(raw, actions, self.max_chars)


# ---- transitions ------------------------------------------------------------
def make_row(goal: str, observation: str, inventory: str, history: Sequence[str],
             actions: Sequence[str], action: str, next_observation: str,
             next_inventory: str, reward: float, done: bool,
             episode: str = "") -> Dict[str, Any]:
    return {"episode": episode, "goal": goal, "observation": _norm(observation)[:_ROW_CHARS],
            "inventory": _norm(inventory)[:300],
            "history": [str(h)[:300] for h in list(history)[-_HISTORY_TURNS:]],
            "actions": list(actions), "action": action,
            "next_observation": _norm(next_observation)[:_ROW_CHARS],
            "next_inventory": _norm(next_inventory)[:300],
            "reward": float(reward or 0.0), "done": bool(done)}


def outcome_view(rows: Sequence[Dict[str, Any]], this_episode: int = 0,
                 limit: int = 200) -> List[Dict[str, Any]]:
    """What predict() receives as ``outcomes``: the last ``limit`` rows, most
    recent first; ``this_episode`` of them (the newest) are from this episode."""
    rows = list(rows)[-limit:]
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(reversed(rows)):
        a = actual(row)
        out.append({"action": row.get("action", ""),
                    "before": _norm(row.get("observation"))[:_VIEW_CHARS],
                    "after": a["observation"], "same": a["same"],
                    "reward": float(row.get("reward") or 0.0), "ends": a["ends"],
                    "inventory": a["inventory"],
                    "episode": "this" if i < this_episode else "earlier"})
    return out


class TransitionBuffer:
    """Executed transitions of one task from train episodes (append at episode end)."""

    def __init__(self, path: Path, capacity: int = 400):
        self.path = Path(path)
        self.capacity = max(1, int(capacity))

    def load(self) -> List[Dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
        return rows[-self.capacity:]

    def extend(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        kept = (self.load() + list(rows))[-self.capacity:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                       encoding="utf-8")
        os.replace(tmp, self.path)


def replay_sample(rows: Sequence[Dict[str, Any]], limit: int, seed: str = "") -> List[int]:
    """Indices of up to ``limit`` rows, spread over action verbs (first word)."""
    by_verb: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        verb = (str(row.get("action") or "").split() or [""])[0]
        by_verb.setdefault(verb, []).append(i)
    rng = random.Random(seed)
    for idx in by_verb.values():
        rng.shuffle(idx)
    picked: List[int] = []
    while len(picked) < limit and any(by_verb.values()):
        for verb in sorted(by_verb):
            if by_verb[verb] and len(picked) < limit:
                picked.append(by_verb[verb].pop())
    return sorted(picked)


def score(model: WorldModel, rows: Sequence[Dict[str, Any]], indices: Sequence[int],
          memory: int = 200) -> Dict[str, Any]:
    """Accuracy of ``model`` on the executed action of ``rows[i]`` for each index.

    Each row sees only the rows before it as ``outcomes`` (no peeking at its
    own result).  Returns coverage, accuracy, errors and the wrong predictions.
    """
    checked = hits = dumps = 0
    errors: List[str] = []
    wrong: List[Dict[str, Any]] = []
    for i in indices:
        row = rows[i]
        try:
            preds = model.predict(row.get("goal", ""), row.get("observation", ""),
                                  row.get("inventory", ""), row.get("history") or [],
                                  row.get("actions") or [row.get("action", "")],
                                  outcome_view(rows[max(0, i - memory):i], limit=memory))
        except SandboxError as exc:
            errors.append(str(exc))
            continue
        dumps += sum(1 for p in preds.values() if p["effect"].lstrip()[:1] in ("{", "["))
        pred = preds.get(row.get("action", ""))
        ok, failed = check(pred, row.get("observation", ""), row.get("next_observation", ""),
                           row.get("next_inventory", ""), row.get("reward", 0.0),
                           row.get("done", False))
        if ok is None:
            continue
        checked += 1
        hits += int(ok)
        if not ok:
            wrong.append({"index": i, "action": row.get("action", ""), "predicted": pred,
                          "failed": failed, "before": _norm(row.get("observation"))[:400],
                          "inventory": _norm(row.get("inventory"))[:150],
                          "actual": actual(row)})
    n = len(indices)
    return {"n": n, "checked": checked, "hits": hits, "errors": errors, "wrong": wrong,
            "dumps": dumps,
            "coverage": checked / n if n else 0.0,
            "accuracy": hits / checked if checked else 0.0}


__all__ = ["FIELDS", "WorldModel", "TransitionBuffer", "actual", "check", "make_row",
           "normalize", "outcome_view", "replay_sample", "score"]
