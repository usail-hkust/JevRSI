"""State memory: a System-2-written extractor plus a framework-kept graph.

The program is deliberately tiny -- ``state(goal, observation, inventory,
history)`` returns one exploration state, e.g. a room name, an event name, or a
small dict.  It does not maintain the graph; the framework does, so every state
program produces the same graph shape and System-2 never has to reason about
storage.  Nodes are states (``room: kitchen``, ``event: picked apple``), edges
are ``node --action--> node`` transitions seen in earlier episodes.

The graph is persisted per (skill, task) under the skill store and summarised
into System-1's prompt.  Programs run in a small, dependency-free sandbox.
"""
from __future__ import annotations

import ast
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAX_LABEL_CHARS = 240

_ALLOWED_AST = {
    ast.Module, ast.FunctionDef, ast.Lambda, ast.arguments, ast.arg, ast.Return,
    ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.If, ast.For,
    ast.While, ast.Break, ast.Continue, ast.Pass, ast.Try, ast.ExceptHandler,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Constant, ast.Name, ast.Load, ast.Store, ast.Del,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice,
    ast.Call, ast.keyword, ast.Attribute, ast.JoinedStr, ast.FormattedValue,
    ast.IfExp, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.comprehension, ast.Starred,
    ast.And, ast.Or, ast.Not, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,
}
_FORBIDDEN_NAMES = {
    "eval", "exec", "open", "compile", "input", "breakpoint", "globals",
    "locals", "vars", "dir", "getattr", "setattr", "delattr", "help",
    "__import__", "memoryview", "object", "type", "super",
}
_SAFE_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "min": min, "max": max, "sum": sum, "round": round, "abs": abs,
    "sorted": sorted, "reversed": reversed, "enumerate": enumerate,
    "zip": zip, "range": range, "any": any, "all": all, "isinstance": isinstance,
    "json": json, "math": math, "re": re,
}


class SandboxError(ValueError):
    """The program is invalid or could not be compiled."""


def _check_ast(tree: ast.AST) -> None:
    if not isinstance(tree, ast.Module) or not tree.body:
        raise SandboxError("empty program")
    if not all(isinstance(node, ast.FunctionDef) for node in tree.body):
        raise SandboxError("top-level code must be function definitions")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST:
            raise SandboxError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("_") or name in _FORBIDDEN_NAMES:
                raise SandboxError(f"disallowed name: {name}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise SandboxError(f"disallowed attribute: {node.attr}")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and (fn.id.startswith("_") or fn.id in _FORBIDDEN_NAMES):
                raise SandboxError(f"disallowed call: {fn.id}")


def _normalise_value(value: Any) -> Tuple[str, Any]:
    """(node label, stored data) from whatever the program returned."""
    if value is None:
        return "(unknown)", None
    if isinstance(value, str):
        label = " ".join(value.split())[:MAX_LABEL_CHARS] or "(unknown)"
        return label, label
    try:
        data = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError):
        return "(unknown)", None
    label = json.dumps(data, ensure_ascii=False, sort_keys=True)[:MAX_LABEL_CHARS]
    return label, data


class StateProgram:
    """A compiled ``state(goal, observation, inventory, history)`` function."""

    def __init__(self, code: str, fn, budget_ms: float = 50.0, max_lines: int = 20000):
        self.code = code
        self.fn = fn
        self.budget_ms = max(1.0, float(budget_ms))
        self.max_lines = max(100, int(max_lines))

    @classmethod
    def compile(cls, code: str, max_chars: int = 6000, budget_ms: float = 50.0,
                max_lines: int = 20000, entry: str = "state") -> "StateProgram":
        code = (code or "").strip()
        if not code:
            raise SandboxError("empty state program")
        if len(code) > max_chars:
            raise SandboxError(f"state program is {len(code)} chars (limit {max_chars})")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise SandboxError(f"syntax error: {exc}") from exc
        _check_ast(tree)
        namespace: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        try:
            exec(compile(tree, "<state_program>", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            raise SandboxError(f"program failed to load: {exc}") from exc
        fn = namespace.get(entry)
        if not callable(fn):
            raise SandboxError(f"the program must define {entry}(goal, observation, inventory, history)")
        return cls(code, fn, budget_ms, max_lines)

    def _run(self, goal: str, observation: str, inventory: str,
             history: Optional[List[str]] = None) -> Any:
        return self._traced(self.fn, goal or "", observation or "", inventory or "",
                            list(history or []))

    def _traced(self, fn, *args) -> Any:
        """Call ``fn`` under the time and line budget of this program."""
        deadline = time.monotonic() + self.budget_ms / 1000.0
        line_count = 0

        def trace(frame, event, arg):
            nonlocal line_count
            if event == "line":
                line_count += 1
                if line_count > self.max_lines:
                    raise SandboxError("state program exceeded its line budget")
                if time.monotonic() > deadline:
                    raise SandboxError("state program exceeded its time budget")
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            return fn(*args)
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"state program error: {exc}") from exc
        finally:
            sys.settrace(previous)

    def __call__(self, goal: str, observation: str, inventory: str,
                 history: Optional[List[str]] = None) -> Tuple[str, Any]:
        """Run one extraction; raises ``SandboxError`` on failure or timeout."""
        return _normalise_value(self._run(goal, observation, inventory, history))

    def holds(self, goal: str, observation: str, inventory: str,
              history: Optional[List[str]] = None) -> bool:
        """Run a predicate (a procedure's precondition); raises ``SandboxError``."""
        return bool(self._run(goal, observation, inventory, history))


@dataclass
class StateGraph:
    """What earlier episodes saw: nodes (states) and action-labelled edges."""

    skill: str
    task: str = ""
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current: str = ""

    @staticmethod
    def node_id(label: str) -> str:
        return " ".join(str(label or "").split()).lower()[:MAX_LABEL_CHARS]

    def observe(self, label: str, action: Optional[str] = None, step: int = 0) -> str:
        node = self.node_id(label)
        if not node:
            return self.current
        info = self.nodes.setdefault(node, {"label": label, "visits": 0, "last_step": 0})
        info["label"] = label
        info["visits"] = int(info.get("visits", 0)) + 1
        info["last_step"] = int(step)
        if action and self.current:
            key = f"{self.current}\x1f{action}"
            edge = self.edges.setdefault(key, {
                "from": self.current, "action": action, "to": node,
                "count": 0, "last_step": 0,
            })
            edge["count"] = int(edge.get("count", 0)) + 1
            edge["last_step"] = int(step)
        self.current = node
        return node

    def summary(self, max_nodes: int = 12, view_chars: int = 1200) -> str:
        if not self.nodes:
            return ""
        lines: List[str] = []
        cur_info = self.nodes.get(self.current)
        if cur_info:
            lines.append(f"current: {cur_info['label']}")
        recent = sorted(self.nodes.values(), key=lambda n: -int(n.get("last_step", 0)))
        labels = [n["label"] for n in recent[:max_nodes]]
        lines.append("seen: " + "; ".join(labels))
        outgoing = [e for e in self.edges.values() if e.get("from") == self.current]
        outgoing.sort(key=lambda e: (-int(e.get("count", 0)), e.get("action", "")))
        if outgoing:
            lines.append("from here: " + "; ".join(
                f"{e['action']} -> {self.nodes.get(e['to'], {}).get('label', e['to'])}"
                for e in outgoing[:max_nodes]))
        text = "\n".join(lines)
        return text if len(text) <= view_chars else text[: view_chars - 3] + "..."

    def transitions(self) -> Dict[str, Dict[str, Any]]:
        """{action: {"next": state label, "count": n}} seen from the current state."""
        out: Dict[str, Dict[str, Any]] = {}
        for e in self.edges.values():
            if e.get("from") != self.current:
                continue
            nxt = self.nodes.get(e.get("to"), {}).get("label", e.get("to"))
            prev = out.get(e.get("action"))
            if prev is None or int(e.get("count", 0)) > int(prev["count"]):
                out[e.get("action")] = {"next": nxt, "count": int(e.get("count", 0))}
        return out

    # ---- persistence -----------------------------------------------------
    @classmethod
    def load(cls, path: Path, skill: str, task: str = "") -> "StateGraph":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(skill=skill, task=task,
                       nodes=dict(data.get("nodes") or {}),
                       edges=dict(data.get("edges") or {}),
                       current=str(data.get("current") or ""))
        except (OSError, ValueError, TypeError):
            return cls(skill=skill, task=task)

    def save(self, path: Path) -> None:
        payload = {"skill": self.skill, "task": self.task, "current": self.current,
                   "nodes": self.nodes, "edges": self.edges}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{path.suffix}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
