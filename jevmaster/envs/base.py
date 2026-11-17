"""The environment adapter interface.

Every benchmark (ALFWorld, and future text-craft / Atari /
Crafter) is wrapped in an :class:`EnvAdapter` so the agent only ever sees a
normalized :class:`Observation` and a list of legal :class:`Candidate` actions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from jevmaster.types import Candidate, Observation, StepResult


class EnvAdapter(ABC):
    """Normalized, episodic environment interface."""

    name: str = "base"
    # The scale ``score()`` is reported on, so a partial score can be read as a
    # fraction of the task: the effect ledger credits partial progress, which
    # is how a losing plan shows up before it costs an episode outright.  An
    # environment whose score is already 0/1 keeps the default.
    score_max: float = 1.0

    @abstractmethod
    def task_variations(self) -> List[Tuple[str, int]]:
        """Return the (task_name, variation_id) pairs selected by the config."""

    @abstractmethod
    def reset(self, task: str, variation: int) -> Observation:
        """Load a specific task/variation and return the initial observation."""

    @abstractmethod
    def candidates(self) -> List[Candidate]:
        """Return the legal actions available in the CURRENT state.

        Contract: the action space is dynamic -- it must be a function of the
        current environment state and may change after every step (e.g.
        ALFWorld's admissible
        commands). Agents re-query this every step and treat the returned list
        as the ONLY legal actions, so it must be explicit and complete for the
        current state.
        """

    @abstractmethod
    def step(self, action: str) -> StepResult:
        """Apply ``action`` and return the outcome."""

    def score(self) -> float:
        """Return the current task score/progress if the env exposes one."""
        return 0.0

    def close(self) -> None:
        """Release any underlying resources."""
        return None
