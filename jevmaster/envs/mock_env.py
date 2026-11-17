"""A tiny deterministic environment for tests and offline smoke runs.

It is NOT a dataset -- just enough to exercise the full agent loop (candidates,
stepping, reward, done) without installing Java / ALFWorld.

The task: reach the goal number by repeatedly choosing the action that moves a
counter toward a target. The "correct" action each step is ``inc`` until the
counter hits the target, then ``finish``.
"""
from __future__ import annotations

from typing import List, Tuple

from jevmaster.envs.base import EnvAdapter
from jevmaster.envs.registry import register_env
from jevmaster.types import Candidate, Observation, StepResult


class MockEnv(EnvAdapter):
    name = "mock"

    def __init__(self, target: int = 3, step_limit: int = 20, num_tasks: int = 1, **_):
        self.target = target
        self.step_limit = step_limit
        self.num_tasks = num_tasks
        self._counter = 0
        self._step = 0
        self._done = False
        self._task = "count-to-target"

    def task_variations(self) -> List[Tuple[str, int]]:
        return [(self._task, i) for i in range(self.num_tasks)]

    def reset(self, task: str, variation: int) -> Observation:
        self._counter = 0
        self._step = 0
        self._done = False
        self._task = task
        return self._observe()

    def _observe(self) -> Observation:
        return Observation(
            task=self._task,
            goal=f"increase the counter to {self.target} then finish",
            text=f"counter is {self._counter}",
            inventory="",
            step=self._step,
            extra={"counter": self._counter, "target": self.target},
        )

    def candidates(self) -> List[Candidate]:
        # Dynamic, state-dependent action space (mirrors the real benchmarks):
        # `dec` is a no-op at 0, so it is simply not offered there.
        out = [Candidate("inc")]
        if self._counter > 0:
            out.append(Candidate("dec"))
        out.append(Candidate("finish"))
        return out

    def step(self, action: str) -> StepResult:
        self._step += 1
        reward = 0.0
        if action == "inc":
            self._counter += 1
        elif action == "dec":
            self._counter = max(0, self._counter - 1)
        elif action == "finish":
            if self._counter == self.target:
                reward = 1.0
            self._done = True
        if self._step >= self.step_limit:
            self._done = True
        return StepResult(
            observation=self._observe(),
            reward=reward,
            done=self._done,
            info={"counter": self._counter},
        )

    def score(self) -> float:
        return 1.0 if self._counter == self.target else 0.0


@register_env("mock")
def _make_mock(**kwargs) -> MockEnv:
    return MockEnv(**kwargs)
