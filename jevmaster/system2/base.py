"""System-2 interface: one judge turn and one develop turn."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union

from jevmaster.skills.skill import Skill
from jevmaster.types import ReviewJob, ReviewResult

# skill name -> the skill (or its text); None when it does not exist
SkillLookup = Callable[[str], Optional[Union[Skill, str]]]


class System2(ABC):
    backend: str = "base"

    @abstractmethod
    def review(self, job: ReviewJob, shown_blocks: Dict[str, str],
               existing_blocks: Dict[str, str], lookup: SkillLookup,
               blocked_blocks: Optional[Dict[str, str]] = None,
               check: Optional[Callable[[Skill], Optional[str]]] = None,
               max_fixes: int = 1, features: Optional[dict] = None) -> ReviewResult:
        """Judge the uncertain steps; develop at most one component."""

    def close(self) -> None:
        return None
