"""The one skill file format shared by the three M5 components.

A skill is a markdown file with YAML frontmatter and a body whose meaning
depends on ``kind``:

* ``procedure`` -- experience memory: a short "when X, do Y" rule.
* ``state``     -- a small program that reduces each observation to one
  exploration state (a room, an event, a region, ...).  The framework builds
  a graph of the states it has visited; the program only names the state.
* ``world_model`` -- (M5.3) a program predicting what each legal action does
  now; its effect texts are shown next to the actions and every prediction
  for the executed action is checked against what happened.

Utility is tracked per revision: ``uses`` / ``successes`` / ``utility`` and
``failure_streak``.  Three consecutive failures set ``blocked``; a revision
resets the counters and unblocks the skill.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

KINDS = ("procedure", "state", "world_model")

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_PY_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text or "skill"


def _scalar(value: str):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [v.strip().strip("'\"") for v in inner.split(",") if v.strip()] if inner else []
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _lenient_frontmatter(text: str) -> dict:
    """Line-based fallback for model-written frontmatter that is not strict YAML."""
    meta: Dict[str, object] = {}
    key = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            if key is not None:
                meta[key] = f"{meta.get(key, '')} {line.strip()}".strip()
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            meta[key] = _scalar(value)
    return meta


@dataclass
class Skill:
    name: str
    kind: str = "procedure"
    when: str = ""                  # retrieval field: the situation it applies to
    keywords: List[str] = field(default_factory=list)
    expected: str = ""              # observable result, used by System-2's judge
    env: str = ""
    tasks: List[str] = field(default_factory=list)
    # Goals of the episodes this skill was developed from (set by the
    goals: List[str] = field(default_factory=list)
    body: str = ""                  # procedure text / state program / world-model program
    # Patterns the runtime must not offer to System-1 (structured, never prose).
    avoid: List[str] = field(default_factory=list)
    # per-revision utility
    uses: int = 0
    successes: int = 0
    utility: float = 0.0
    # In relative mode gain is success - task baseline, not raw success.
    gain_sum: float = 0.0
    validated: bool = False
    failure_streak: int = 0
    blocked: bool = False
    # world_model: live predictions checked / correct (this revision) and the
    # accuracy measured on recorded transitions when it was admitted.
    checks: int = 0
    hits: int = 0
    replay_accuracy: float = 0.0
    replay_n: int = 0
    revision: int = 1
    created: str = field(default_factory=now_iso)
    updated: str = field(default_factory=now_iso)
    source: str = "system2"
    trigger_context: str = ""
    path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.kind = str(self.kind or "procedure").lower()
        if self.kind not in KINDS:
            self.kind = "procedure"

    # ---- serialization ---------------------------------------------------
    def frontmatter(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "when": self.when,
            "keywords": list(self.keywords),
            "expected": self.expected,
            "env": self.env,
            "tasks": list(self.tasks),
            "goals": list(self.goals),
            "avoid": list(self.avoid),
            "uses": int(self.uses),
            "successes": int(self.successes),
            "utility": round(float(self.utility), 4),
            "gain": round(float(self.gain), 4),
            "gain_sum": round(float(self.gain_sum), 4),
            "validated": bool(self.validated),
            "failure_streak": int(self.failure_streak),
            "blocked": bool(self.blocked),
            "checks": int(self.checks),
            "hits": int(self.hits),
            "replay_accuracy": round(float(self.replay_accuracy), 4),
            "replay_n": int(self.replay_n),
            "revision": int(self.revision),
            "created": self.created,
            "updated": self.updated,
            "source": self.source,
            "trigger_context": self.trigger_context,
        }

    def to_markdown(self) -> str:
        fm = yaml.safe_dump(self.frontmatter(), sort_keys=False, allow_unicode=True)
        body = self.body.rstrip()
        return f"---\n{fm}---\n\n{body}\n" if body else f"---\n{fm}---\n"

    @classmethod
    def from_markdown(cls, text: str, path: Optional[Path] = None) -> "Skill":
        match = _FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError("skill markdown is missing a YAML frontmatter block")
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = _lenient_frontmatter(match.group(1))
        if not isinstance(meta, dict):
            meta = _lenient_frontmatter(match.group(1))
        if not meta:
            raise ValueError("skill frontmatter is not a mapping")
        keywords = meta.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        tasks = meta.get("tasks") or []
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",") if t.strip()]
        goals = meta.get("goals") or []
        if isinstance(goals, str):
            goals = [goals]
        avoid = meta.get("avoid") or []
        if isinstance(avoid, str):
            avoid = [a.strip() for a in avoid.split(",") if a.strip()]
        return cls(
            name=str(meta.get("name") or (path.stem if path else "unnamed")),
            kind=str(meta.get("kind") or "procedure"),
            when=str(meta.get("when") or ""),
            keywords=[str(k) for k in keywords],
            expected=str(meta.get("expected") or ""),
            env=str(meta.get("env") or ""),
            tasks=[str(t) for t in tasks],
            goals=[str(g) for g in goals if str(g).strip()],
            body=match.group(2).strip(),
            avoid=[str(a) for a in avoid if str(a).strip()],
            uses=int(meta.get("uses") or 0),
            successes=int(meta.get("successes") or 0),
            utility=float(meta.get("utility") or 0.0),
            gain_sum=float(meta.get("gain_sum") or 0.0),
            validated=bool(meta.get("validated") or False),
            failure_streak=int(meta.get("failure_streak") or 0),
            blocked=bool(meta.get("blocked") or False),
            checks=int(meta.get("checks") or 0),
            hits=int(meta.get("hits") or 0),
            replay_accuracy=float(meta.get("replay_accuracy") or 0.0),
            replay_n=int(meta.get("replay_n") or 0),
            revision=int(meta.get("revision") or 1),
            created=str(meta.get("created") or now_iso()),
            updated=str(meta.get("updated") or now_iso()),
            source=str(meta.get("source") or "system2"),
            trigger_context=str(meta.get("trigger_context") or ""),
            path=path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Skill":
        path = Path(path)
        return cls.from_markdown(path.read_text(encoding="utf-8"), path=path)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save the skill")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        tmp.write_text(self.to_markdown(), encoding="utf-8")
        os.replace(tmp, target)
        self.path = target
        return target

    # ---- content accessors ----------------------------------------------
    def retrieval_body(self) -> str:
        """Body text for BM25; code blocks are implementation, not semantics."""
        return _CODE_BLOCK_RE.sub(" ", self.body)

    def text_for_retrieval(self) -> str:
        return " ".join([self.name, " ".join(self.keywords), self.when,
                         self.expected, self.retrieval_body()]).lower()

    def text_for_match(self) -> str:
        return self.text_for_retrieval()

    def program_code(self) -> str:
        """The state program: the ``## Program`` block (or the first python block)."""
        section = self.body
        marker = re.search(r"^##\s*Program\s*$", section, re.IGNORECASE | re.MULTILINE)
        if marker:
            section = section[marker.end():]
        match = _PY_BLOCK_RE.search(section)
        return match.group(1).strip() if match else ""

    # ---- prompt rendering / utility -------------------------------------
    def advice_body(self) -> str:
        """The body System-1 sees."""
        return self.body

    def as_prompt_block(self) -> str:
        header = f"- {self.name}"
        if self.when:
            header += f" (use when: {self.when})"
        body = self.advice_body().strip()
        return f"{header}\n{body}" if body else header

    def as_review_block(self, show_gain: bool = False) -> str:
        lines = [self.as_prompt_block()]
        if self.expected:
            lines.append(f"expected: {self.expected}")
        if self.avoid:
            lines.append(f"avoid: {', '.join(self.avoid)}")
        if self.kind == "world_model":
            lines.append(f"record: revision {self.revision}, live accuracy "
                         f"{self.live_accuracy:.0%} ({self.hits}/{self.checks} checked predictions), "
                         f"accuracy on recorded transitions {self.replay_accuracy:.0%} "
                         f"(n={self.replay_n})")
            return "\n".join(lines)
        record = (f"record: revision {self.revision}, uses {self.uses}, "
                  f"successes {self.successes}, utility {self.utility:.2f}, "
                  f"failure streak {self.failure_streak}")
        if show_gain:
            record += f", gain {self.gain:.2f}"
            if self.validated:
                record += ", VALIDATED"
        if self.blocked:
            record += ", BLOCKED"
        lines.append(record)
        return "\n".join(lines)

    @property
    def live_accuracy(self) -> float:
        return self.hits / self.checks if self.checks else 0.0

    @property
    def gain(self) -> float:
        return self.gain_sum / self.uses if self.uses else 0.0

    def record_outcome(self, success: bool, block_after: int = 3, *,
                       baseline: Optional[float] = None,
                       relative: bool = False,
                       min_uses: int = 3, min_gain: float = 0.0,
                       progress: Optional[float] = None) -> bool:
        """Update per-revision utility; return True when this call blocks the skill.

        Absolute mode keeps M5 semantics (three consecutive failures).  Relative
        mode credits success against the task baseline and blocks only a skill
        that has enough trials and no positive gain.
        """
        self.uses += 1
        if success:
            self.successes += 1
            self.failure_streak = 0
        else:
            self.failure_streak += 1
        self.utility = self.successes / self.uses if self.uses else 0.0
        if relative and baseline is not None:
            value = float(progress) if progress is not None else float(bool(success))
            self.gain_sum += value - float(baseline)
            enough = self.uses >= max(1, int(min_uses))
            self.validated = bool(enough and self.gain > float(min_gain))
            if enough and self.gain <= float(min_gain):
                self.blocked = True
        self.updated = now_iso()
        if not (relative and baseline is not None):
            if self.failure_streak >= max(1, int(block_after)):
                self.blocked = True
        return self.blocked

    def reset_revision(self) -> None:
        """A revision is a clean slate: counters, gain and the block are cleared."""
        self.revision += 1
        self.uses = 0
        self.successes = 0
        self.utility = 0.0
        self.gain_sum = 0.0
        self.validated = False
        self.failure_streak = 0
        self.blocked = False
        self.checks = 0
        self.hits = 0
        self.updated = now_iso()
