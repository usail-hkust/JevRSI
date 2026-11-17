"""Deterministic System-2 for offline smoke tests and CI.

Runs the real judge/develop protocol against a scripted chat.  ``judge``
selects the outcome: ``auto`` (NEW procedure when no procedure was shown),
``new``, ``none``, ``state`` or ``procedure``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from jevmaster.skills.retrieval import tokenize
from jevmaster.skills.skill import slugify
from jevmaster.prompts.system2 import WORLD_SYSTEM_PROMPT
from jevmaster.system2.base import SkillLookup, System2
from jevmaster.system2.clients.openai_client import ChatResult
from jevmaster.system2.llm import run_review
from jevmaster.types import ReviewJob, ReviewResult


# A world model that only repeats what the same action did from the same observation.
STUB_WORLD_MODEL = """```markdown
---
name: repeat-last-outcome
kind: world_model
when: any situation already seen in this task
keywords: [repeat]
---
## Rules
- the same action from the same observation does what it did last time
## Program
```python
def predict(goal, observation, inventory, history, actions, outcomes):
    now = " ".join(observation.split())[:300]
    out = {}
    for o in outcomes:
        a = o["action"]
        if a in actions and a not in out and o["before"] == now:
            out[a] = {"effect": "nothing changes" if o["same"] else "as last time",
                      "same": o["same"], "ends": o["ends"]}
    return out
```
```"""


class StubSystem2(System2):
    backend = "stub"

    def __init__(self, judge: str = "auto", **_):
        self.judge = str(judge or "auto").lower()
        self._kind = "procedure"

    # ---- scripted turns --------------------------------------------------
    def _judge(self, job: ReviewJob, shown_blocks: Dict[str, str]) -> str:
        mode = self.judge
        if mode == "none":
            return "REASON: the decision worked.\nDECISION: NONE"
        if mode == "state":
            self._kind = "state"
            return ("REASON: S1 forgets where it is.\nDECISION: NEW state"
                    if not job.state_program else
                    "REASON: state is already tracked.\nDECISION: NONE")
        if mode == "procedure":
            self._kind = "procedure"
            return "REASON: no rule covered this.\nDECISION: NEW procedure"
        if shown_blocks:
            return "REASON: a procedure covered this.\nDECISION: NONE"
        self._kind = "procedure"
        return "REASON: no rule covered this.\nDECISION: NEW procedure"

    def _develop(self, job: ReviewJob, kind: str) -> str:
        goal_tokens = tokenize(job.goal)[:3]
        name = f"handle-{'-'.join(goal_tokens) or slugify(job.task) or 'task'}"
        keywords = list(dict.fromkeys(tokenize(job.goal) + tokenize(job.observation)))[:6]
        when = "goal involves " + (" ".join(goal_tokens) or "this task")
        if kind == "state":
            return (
                "```markdown\n---\n"
                f"name: track-{slugify(job.task) or 'state'}\nkind: state\n"
                f"when: {when}\nkeywords: [{', '.join(keywords)}]\n---\n"
                "## State\n- label: a short state derived from the observation\n"
                "## Program\n```python\n"
                "def state(goal, observation, inventory, history):\n"
                "    return ' '.join(observation.split())[:80] or 'unknown'\n"
                "```\n```"
            )
        action = job.trajectory[-1]["action"] if job.trajectory else "look around"
        return (
            "```markdown\n---\n"
            f"name: {name}\nkind: procedure\n"
            f"when: {when}\nkeywords: [{', '.join(keywords)}]\n"
            "expected: the goal makes progress within a few steps\n---\n"
            "## Procedure\n"
            f"1. If the goal is unmet, prefer actions like '{action}'.\n"
            "2. If the goal is satisfied, prefer a finishing action.\n```"
        )

    def review(self, job: ReviewJob, shown_blocks: Dict[str, str],
               existing_blocks: Dict[str, str], lookup: SkillLookup,
               blocked_blocks: Optional[Dict[str, str]] = None,
               check=None, max_fixes: int = 1, features=None) -> ReviewResult:
        def chat(messages):
            if messages and messages[0].get("content") == WORLD_SYSTEM_PROMPT:
                return ChatResult(text=STUB_WORLD_MODEL)
            if len(messages) <= 2:
                self._kind = "procedure"
                return ChatResult(text=self._judge(job, shown_blocks))
            return ChatResult(text=self._develop(job, self._kind))

        return run_review(chat, job, shown_blocks, existing_blocks, lookup,
                          blocked_blocks=blocked_blocks, check=check,
                          max_fixes=max_fixes, features=features)
