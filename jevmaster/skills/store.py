"""File-backed skill store (one markdown file per skill).

System-2 is the only writer; System-1 holds a read-only store over the same
directory and calls :meth:`SkillStore.refresh` before retrieval.  Every write
rewrites ``.version`` so the other process can detect changes cheaply.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from jevmaster.skills.skill import Skill, now_iso, slugify

_VERSION_FILE = ".version"
_GRAPH_DIR = "_graphs"
_WORLD_DIR = "_world"           # world-model transition buffers, one file per task
_HISTORY_DIR = "_history"
_BASELINE_FILE = "_baselines.json"
_FATAL_FILE = "_fatal_rules.json"
_MAX_GOALS = 8                 # source goals kept per skill (most recent)


class SkillStore:
    def __init__(self, root: str | Path, env_name: str, frozen: bool = False,
                 block_after: int = 3, lifecycle: Optional[dict] = None,
                 blocking: bool = True):
        self.root = Path(root)
        self.env_name = env_name
        self.dir = self.root / env_name
        self.frozen = bool(frozen)
        self.block_after = max(1, int(block_after))
        # M5.3: failure streaks can be switched off entirely (kept for runs that
        # compare admission rules); the default keeps the M5 rule.
        self.blocking = bool(blocking)
        lc = lifecycle or {}
        self.lifecycle_mode = str(lc.get("mode", "absolute") or "absolute").lower()
        self.min_baseline_trials = max(1, int(lc.get("min_baseline_trials", 2)))
        self.min_uses = max(1, int(lc.get("min_uses", 3)))
        self.min_gain = float(lc.get("min_gain", 0.0))
        self.use_progress = bool(lc.get("progress", False))
        self.snapshot_require_validated = bool(lc.get("snapshot_require_validated", False))
        self.snapshot_fallback_top_k = max(0, int(lc.get("snapshot_fallback_top_k", 3)))
        self.max_history = max(1, int(lc.get("max_history", 8)))
        self._lock = threading.RLock()
        self._skills: Dict[str, Skill] = {}
        self._version = 0
        self._disk_stamp: Optional[str] = None
        self._baselines: Dict[str, Any] = self._load_baselines()
        self._load_from_disk()

    # ---- loading ---------------------------------------------------------
    def _read_stamp(self) -> Optional[str]:
        try:
            return (self.dir / _VERSION_FILE).read_text(encoding="utf-8")
        except OSError:
            return None

    def _load_from_disk(self) -> None:
        with self._lock:
            self._skills.clear()
            self._disk_stamp = self._read_stamp()
            self.dir.mkdir(parents=True, exist_ok=True)
            for md in sorted(self.dir.glob("*.md")):
                try:
                    skill = Skill.load(md)
                except Exception:
                    continue        # a malformed file never crashes a run
                self._skills[md.stem] = skill
            self._version += 1

    def refresh(self) -> bool:
        stamp = self._read_stamp()
        with self._lock:
            if stamp == self._disk_stamp:
                return False
        self._load_from_disk()
        return True

    def _bump(self) -> None:
        self._version += 1
        stamp = f"{time.time_ns()}-{os.getpid()}-{self._version}"
        tmp = self.dir / f"{_VERSION_FILE}.{os.getpid()}.tmp"
        tmp.write_text(stamp, encoding="utf-8")
        os.replace(tmp, self.dir / _VERSION_FILE)
        self._disk_stamp = stamp

    # ---- baseline / history files ---------------------------------------
    def _load_baselines(self) -> Dict[str, Any]:
        try:
            raw = json.loads((self.dir / _BASELINE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault("global", {"n": 0, "successes": 0, "progress_sum": 0.0})
        raw.setdefault("tasks", {})
        raw.setdefault("goals", {})
        return raw

    def _save_baselines(self) -> None:
        tmp = self.dir / f".{_BASELINE_FILE}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(self._baselines, ensure_ascii=False, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self.dir / _BASELINE_FILE)

    @staticmethod
    def _goal_key(goal: str) -> str:
        return " ".join(str(goal or "").lower().split())[:200]

    def observe_baseline(self, task: str, goal: str, success: bool,
                         progress: Optional[float] = None) -> None:
        """Record a no-skill (or ablation) episode in the baseline posterior."""
        if self.frozen:
            return
        with self._lock:
            for bucket, key in ((self._baselines["global"], None),
                                (self._baselines["tasks"], str(task or "")),
                                (self._baselines["goals"], self._goal_key(goal))):
                if key is not None and not key:
                    continue
                target = bucket if key is None else bucket.setdefault(
                    key, {"n": 0, "successes": 0})
                target["n"] = int(target.get("n", 0)) + 1
                target["successes"] = int(target.get("successes", 0)) + int(bool(success))
                target["progress_sum"] = float(target.get("progress_sum", 0.0)) + float(
                    progress if progress is not None else bool(success))
            self._save_baselines()

    def baseline_for(self, task: str, goal: str,
                     min_trials: Optional[int] = None) -> Optional[float]:
        """Smoothed no-skill success rate: exact goal, then task, then global."""
        need = self.min_baseline_trials if min_trials is None else max(1, int(min_trials))
        with self._lock:
            for bucket, key in ((self._baselines.get("goals", {}), self._goal_key(goal)),
                                (self._baselines.get("tasks", {}), str(task or "")),
                                ({"global": self._baselines.get("global", {})}, "global")):
                entry = bucket.get(key) or {}
                n = int(entry.get("n", 0))
                if n >= need:
                    if self.use_progress and "progress_sum" in entry:
                        return float(entry.get("progress_sum", 0.0)) / (n + 1.0)
                    return (int(entry.get("successes", 0)) + 1.0) / (n + 2.0)
        return None

    def _history_dir(self, slug: str) -> Path:
        return self.dir / _HISTORY_DIR / slug

    def _archive_revision(self, slug: str, skill: Skill) -> None:
        target_dir = self._history_dir(slug)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"rev{skill.revision}.md"
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(skill.to_markdown(), encoding="utf-8")
        os.replace(tmp, path)
        files = sorted(target_dir.glob("rev*.md"),
                       key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
        for old in files[:-self.max_history]:
            try:
                old.unlink()
            except OSError:
                pass

    def _history_versions(self, slug: str) -> List[Skill]:
        out: List[Skill] = []
        for path in sorted(self._history_dir(slug).glob("rev*.md"),
                           key=lambda p: int(re.sub(r"\D", "", p.stem) or 0),
                           reverse=True):
            try:
                out.append(Skill.load(path))
            except Exception:
                continue
        return out

    def export_snapshot(self, target: str | Path,
                        require_validated: Optional[bool] = None) -> int:
        """Freeze the train store for eval, optionally keeping only validated skills.

        Active skills are written to ``target``.  If validation is required, an
        unvalidated active revision falls back to the newest validated ancestor
        in ``_history``; if none exists, it is omitted (or kept as a bounded
        fallback when ``snapshot_fallback_top_k`` > 0).
        """
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        require = self.snapshot_require_validated if require_validated is None else bool(require_validated)
        with self._lock:
            selected: Dict[str, Skill] = {}
            fallbacks: List[tuple] = []
            for slug, skill in self._skills.items():
                if not require:
                    selected[slug] = skill
                    continue
                if skill.validated and skill.gain > self.min_gain:
                    selected[slug] = skill
                    continue
                ancestor = next((h for h in self._history_versions(slug)
                                 if h.validated and h.gain > self.min_gain), None)
                if ancestor is not None:
                    selected[slug] = ancestor
                elif not skill.blocked:
                    # Without a validated ancestor, keep the active version:
                    # a brand-new skill has no history and dropping every
                    # unvalidated skill starves eval of the train library.
                    fallbacks.append((skill.uses, skill.gain, slug, skill))
            ranked = sorted(fallbacks, key=lambda row: (row[0], row[1], row[2]), reverse=True)
            if self.snapshot_fallback_top_k > 0:
                ranked = ranked[:self.snapshot_fallback_top_k]
            for _uses, _gain, slug, skill in ranked:
                selected.setdefault(slug, skill)
            for slug, skill in selected.items():
                (target / f"{slug}.md").write_text(skill.to_markdown(), encoding="utf-8")
            for name in (_BASELINE_FILE, _FATAL_FILE, _VERSION_FILE):
                src = self.dir / name
                if src.exists():
                    try:
                        shutil.copy2(src, target / name)
                    except OSError:
                        pass
            for sub in (_GRAPH_DIR, _WORLD_DIR):
                if (self.dir / sub).exists():
                    shutil.copytree(self.dir / sub, target / sub, dirs_exist_ok=True)
            return len(selected)

    # ---- library views ---------------------------------------------------
    def library(self) -> List[Skill]:
        """The skills System-1 can currently be shown (unblocked), by slug."""
        with self._lock:
            return [self._skills[slug] for slug in sorted(self._skills)
                    if not self._skills[slug].blocked]

    @staticmethod
    def library_hash(skills: Sequence[Skill]) -> str:
        """Content identity of a library: what S1 sees, not the counters."""
        h = hashlib.sha1()
        for skill in sorted(skills, key=lambda s: s.name):
            for part in (skill.name, skill.kind, skill.mode, str(skill.revision),
                         skill.when, " ".join(skill.keywords), " ".join(skill.tasks),
                         " ".join(skill.goals), skill.body):
                h.update(part.encode("utf-8", "ignore"))
                h.update(b"\x00")
        return h.hexdigest()[:16]

    def remove(self, name: str) -> bool:
        """Delete a skill file (used for snapshots trimmed by the task gate)."""
        with self._lock:
            slug = self._slug_of(name)
            if slug is None:
                return False
            skill = self._skills.pop(slug)
            try:
                if skill.path is not None:
                    Path(skill.path).unlink()
            except OSError:
                pass
            self._bump()
            return True

    # ---- reads -----------------------------------------------------------
    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def snapshot(self, kind: Optional[str] = None, include_blocked: bool = False,
                 task: Optional[str] = None) -> List[Skill]:
        """Skills of a kind, optionally scoped to ``task``.

        The task scope is a *label*, not the dataset: an eval split may run
        the same environment under another id (MiniGrid OOD is
        ``MixedTestLocal`` while train is ``MixedTrainLocal``).  A store holds
        one environment's skills, so when no skill carries the requested
        label at all, the whole library is returned instead of nothing.
        """
        with self._lock:
            out = []
            for skill in self._skills.values():
                if kind is not None and skill.kind != kind:
                    continue
                if skill.blocked and not include_blocked:
                    continue
                out.append(skill)
            if task is None:
                return out
            scoped = [s for s in out if not s.tasks or task in s.tasks]
            return scoped or out

    def get(self, name: str) -> Optional[Skill]:
        with self._lock:
            for skill in self._skills.values():
                if skill.name == name:
                    return skill
        return None

    def count(self, kind: Optional[str] = None) -> int:
        with self._lock:
            return sum(1 for s in self._skills.values() if kind is None or s.kind == kind)

    def blocked_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._skills.values() if s.blocked)

    def __len__(self) -> int:
        with self._lock:
            return len(self._skills)

    def graph_dir(self, skill_name: str) -> Path:
        return self.dir / _GRAPH_DIR / slugify(skill_name)

    def world_path(self, task: str) -> Path:
        """Train transitions of ``task``, the world model's memory and test set."""
        return self.dir / _WORLD_DIR / f"{slugify(task or 'task')}.jsonl"

    def world_tasks(self) -> List[str]:
        return sorted(p.stem for p in (self.dir / _WORLD_DIR).glob("*.jsonl"))

    def _slug_of(self, name: str) -> Optional[str]:
        for slug, skill in self._skills.items():
            if skill.name == name:
                return slug
        return None

    def _unique_slug(self, name: str) -> str:
        base = slugify(name)
        slug, i = base, 1
        while slug in self._skills:
            i += 1
            slug = f"{base}-{i}"
        return slug

    # ---- writes ----------------------------------------------------------
    def add(self, skill: Skill) -> Optional[Skill]:
        if self.frozen:
            return None
        with self._lock:
            existing = self._slug_of(skill.name)
            if existing is not None:
                return self.revise(skill.name, skill)
            slug = self._unique_slug(skill.name)
            skill.path = self.dir / f"{slug}.md"
            skill.save()
            self._skills[slug] = skill
            self._bump()
            return skill

    def revise(self, name: str, new: Skill) -> Optional[Skill]:
        """Replace content, increment the revision and clear its utility."""
        if self.frozen:
            return None
        with self._lock:
            slug = self._slug_of(name)
            if slug is None:
                return None
            cur = self._skills[slug]
            self._archive_revision(slug, cur)
            cur.kind = new.kind or cur.kind
            cur.when = new.when or cur.when
            cur.keywords = list(new.keywords or cur.keywords)
            cur.expected = new.expected or cur.expected
            cur.body = new.body or cur.body
            cur.tasks = list(dict.fromkeys(cur.tasks + list(new.tasks or [])))
            goals = [g for g in cur.goals if g not in (new.goals or [])]
            cur.goals = (goals + list(new.goals or []))[-_MAX_GOALS:]
            cur.trigger_context = new.trigger_context or cur.trigger_context
            cur.reset_revision()
            cur.save()
            self._bump()
            return cur

    def record_outcomes(self, names: Sequence[str], success: bool,
                        baseline: Optional[float] = None,
                        progress: Optional[float] = None) -> List[Skill]:
        """Credit one episode's outcome to each skill shown/used in it.

        Absolute mode blocks on ``block_after`` consecutive failed episodes.
        Relative mode credits ``success - baseline`` and blocks only after
        ``min_uses`` trials with non-positive mean gain.  Returns the skills
        that became blocked by this call.
        """
        if self.frozen or not names:
            return []
        wanted = set(names)
        blocked: List[Skill] = []
        relative = self.lifecycle_mode == "relative" and baseline is not None
        with self._lock:
            changed = False
            for slug, skill in self._skills.items():
                if skill.name not in wanted:
                    continue
                was = skill.blocked
                skill.record_outcome(
                    success, self.block_after,
                    baseline=baseline, relative=relative,
                    min_uses=self.min_uses, min_gain=self.min_gain,
                    progress=progress if self.use_progress else None)
                if not self.blocking:
                    skill.blocked = was
                skill.save()
                changed = True
                if skill.blocked and not was:
                    blocked.append(skill)
            if changed:
                self._bump()
        return blocked

    def record_predictions(self, name: str, checks: int, hits: int,
                           revision: Optional[int] = None) -> Optional[Skill]:
        """Add live prediction checks of a world model (only its current revision)."""
        if self.frozen or checks <= 0:
            return None
        with self._lock:
            skill = self.get(name)
            if skill is None or (revision is not None and int(revision) != skill.revision):
                return None
            skill.checks += int(checks)
            skill.hits += int(hits)
            skill.updated = now_iso()
            skill.save()
            self._bump()
            return skill

    def set_replay(self, name: str, accuracy: float, n: int) -> Optional[Skill]:
        """The accuracy a world model scored on recorded transitions at admission."""
        if self.frozen:
            return None
        with self._lock:
            skill = self.get(name)
            if skill is None:
                return None
            skill.replay_accuracy, skill.replay_n = float(accuracy), int(n)
            skill.save()
            self._bump()
            return skill

    def unblock(self, name: str) -> bool:
        """Clear the block without changing the content (a fresh chance)."""
        if self.frozen:
            return False
        with self._lock:
            skill = self.get(name)
            if skill is None:
                return False
            skill.blocked = False
            skill.failure_streak = 0
            skill.updated = now_iso()
            skill.save()
            self._bump()
            return True
