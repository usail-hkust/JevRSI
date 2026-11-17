"""System-2 prompts: one judge turn, then one develop turn for one component.

The three components are deliberately small:

* ``procedure`` -- a short markdown rule;
* ``state``     -- a small Python extractor returning one exploration state;

M5.3 adds the ``world_model``: one program per environment predicting what each
legal action does now.  It is not chosen by the judge; System-2 writes it when
the environment has none and revises it whenever a checked prediction was wrong
(:func:`build_world_prompt`).

The judge chooses at most one component, or NONE.  Prompts state each fact
once: trajectory, the three channels System-1 sees, and the existing library.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from jevmaster.skills.skill import KINDS
from jevmaster.types import ReviewJob

DECISION_LINE = "DECISION: NONE | NEW procedure | NEW state | REVISE <name>"

REVIEW_SYSTEM_PROMPT = """You are System-2 of a dual-system agent. System-1 (S1) chooses actions; you never act. S1 was uncertain, and you decide whether one reusable component would help it in future episodes:

- procedure: a short rule for know-how S1 lacks ("when X, do Y"), shown as advice.
- state: a small Python function that names the current exploration state (room, region, event, ...). The framework turns those names into a persistent graph shown to S1.

Choose at most one component. Decline with NONE when S1 recovered, an existing component already covers the situation, or no lesson generalizes beyond this episode. A component must stay valid across episodes and variations: no instance numbers, one-off layouts, or names unique to this episode. Prefer the smallest component that fixes the cause: missing know-how -> procedure; missing episode facts -> state.

Reason in one sentence, then reply with exactly one decision line:
REASON: <one sentence>
""" + DECISION_LINE

JUDGE_TEMPLATE = """Environment: {env}
Task: {task}
Goal: {goal}

Trajectory (" >>" marks the uncertain steps; step | action | S1 confidence | result):
{trajectory}
{status}

State memory at the uncertain step:
{state}

Procedures shown at the uncertain step:
{shown}

{existing_header}
{existing}
{blocked}"""

DEVELOP_COMMON = """Situation at step {step}:
Observation: {observation}
Inventory: {inventory}
Legal actions: {actions}
{target}"""

PROCEDURE_TEMPLATE = DEVELOP_COMMON + """
Write one reusable procedure:
```markdown
---
name: short-kebab-case-name
kind: procedure
when: observable conditions under which this applies
keywords: [k1, k2, k3]
expected: observable result when followed
---
## Procedure
<condition -> decision, for every episode of this task>
```
Reply NONE if nothing reusable can be written."""

STATE_TEMPLATE = DEVELOP_COMMON + """
Write one reusable state extractor:
```markdown
---
name: short-kebab-case-name
kind: state
when: observable conditions under which this applies
keywords: [k1, k2, k3]
---
## State
<one line per returned field>
## Program
```python
def state(goal, observation, inventory, history):
    # pure Python: no imports (re, json, math are available)
    # return one short string (e.g. the room) or a small dict
    return "unknown"
```
```
The function runs after every step. It must derive the state only from its arguments, tolerate unseen texts, and return a JSON-serializable value. Do not keep the graph yourself.

Reply NONE if nothing reusable can be written."""



FIX_TEMPLATE = """Your component failed the admission check:
{problem}
Fix it and reply with the complete component in the same markdown format."""

_DECISION_RE = re.compile(
    r"^[ \t]*DECISION[ \t]*:[ \t]*(NONE|NEW|REVISE)\b[ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE)
_REASON_RE = re.compile(r"^\s*REASON\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([A-Za-z0-9_+-]*)[ \t]*$")
_FRONTMATTER_START = re.compile(r"^---[ \t]*$", re.MULTILINE)


def judge_kinds(features: Optional[dict]) -> List[str]:
    """The component kinds the judge may choose: procedures and state programs.

    A ``features['kinds']`` list can only
    narrow this set, never add a kind.
    """
    kinds = list((features or {}).get("kinds") or ("procedure", "state"))
    return [k for k in ("procedure", "state") if k in kinds]


def review_system_prompt(features: Optional[dict] = None) -> str:
    """The judge's system prompt. The available kinds are fixed (procedure and
    state), so the text no longer varies with ``features``."""
    return REVIEW_SYSTEM_PROMPT


def _clip(text, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def parse_judge(text: str) -> Tuple[str, Optional[str], str, str]:
    """(decision, target, kind, reason) from the last DECISION line."""
    matches = list(_DECISION_RE.finditer(text or ""))
    reason_matches = list(_REASON_RE.finditer(text or ""))
    reason = reason_matches[-1].group(1).strip() if reason_matches else ""
    if not matches:
        return "NONE", None, "procedure", reason or "unparsable judge reply"
    match = matches[-1]
    if not reason:
        reason = " ".join((text or "")[: match.start()].split()).strip()
    decision = match.group(1).upper()
    rest = re.sub(r"[`'\"<>,.]", " ", match.group(2)).strip()
    words = rest.split()
    if decision == "NONE":
        return "NONE", None, "procedure", reason
    if decision == "REVISE":
        target = words[0] if words else None
        return ("REVISE" if target else "NEW"), target, "procedure", reason
    kind = "procedure"
    if words:
        word = words[0].lower()
        if word in ("state", "state_program", "state-program", "program"):
            kind = "state"
        elif word in ("procedure", "skill"):
            kind = "procedure"
    return "NEW", None, kind, reason


def parse_skill_markdown(text: str) -> Optional[str]:
    """Extract one frontmatter item from an LLM reply, with nested code fences.

    The item may sit inside a ```markdown fence.  A state program's inner
    ```python block must not be mistaken for that fence, so fences are tracked
    by depth; a bare fence at depth 0 closes the outer block.  A bare item (no
    outer fence) runs to the end of the reply.
    """
    if not text or text.strip().upper() == "NONE":
        return None
    for start in _FRONTMATTER_START.finditer(text):
        prefix = text[: start.start()].rstrip("\n").splitlines()
        outer = bool(prefix and _FENCE_RE.match(prefix[-1]))
        candidate = text[start.start():]
        lines: List[str] = []
        depth = 0
        for line in candidate.splitlines():
            fence = _FENCE_RE.match(line)
            if fence and outer:
                lang = (fence.group(2) or "").strip()
                if depth == 0:
                    if lang:            # an inner code block opens
                        depth = 1
                        lines.append(line)
                        continue
                    break               # the outer fence closes
                depth -= 1              # an inner code block closes
                lines.append(line)
                continue
            lines.append(line)
        item = "\n".join(lines).strip()
        if item.startswith("---") and item.count("---") >= 2:
            return item
    return None


def _trajectory(job: ReviewJob, feedback_chars: int = 160,
                features: Optional[dict] = None) -> str:
    marked = set(job.uncertain_steps)
    rows = []
    for ev in job.trajectory:
        step = int(ev.get("step", 0))
        mark = ">>" if step in marked else "  "
        action = _clip(ev.get("action", ""), 120)
        if (features or {}).get("world_model") and ev.get("effect"):
            wrong = " -- WRONG" if ev.get("prediction_ok") is False else ""
            action += f" [predicted: {_clip(ev.get('effect'), 80)}{wrong}]"
        if (features or {}).get("progress"):
            reward = ev.get("reward")
            reward_text = f"{float(reward):+.2f}" if reward is not None else "?"
            fatal = bool(
                job.outcome
                and float(job.outcome.get("score") or 0) <= -99.0
                and step == max((int(e.get("step", 0)) for e in job.trajectory), default=-1)
            )
            rows.append(
                f"{mark} {step} | {action} | "
                f"{float(ev.get('confidence', 0.0)):.2f} | "
                f"reward {reward_text}{' | FATAL' if fatal else ''} | "
                f"{_clip(ev.get('feedback', ''), feedback_chars)}")
        else:
            rows.append(
                f"{mark} {step} | {action} | "
                f"{float(ev.get('confidence', 0.0)):.2f} | "
                f"{_clip(ev.get('feedback', ''), feedback_chars)}")
    return "\n".join(rows) or "(none)"


def build_judge_prompt(job: ReviewJob, shown_blocks: Dict[str, str],
                       existing_blocks: Dict[str, str],
                       blocked_blocks: Optional[Dict[str, str]] = None,
                       features: Optional[dict] = None) -> Tuple[str, str]:
    last = int(job.trajectory[-1]["step"]) if job.trajectory else 0
    if job.outcome:
        if (features or {}).get("progress"):
            score = float(job.outcome.get("score") or 0)
            status = (f"The episode ended after step {last}: "
                      f"{'success' if job.outcome.get('success') else 'failure'}"
                      f" (score {score:.1f}"
                      + (", FATAL -100" if score <= -99.0 else "") + ").")
        else:
            status = (f"The episode ended after step {last}: "
                      f"{'success' if job.outcome.get('success') else 'failure'}")
    else:
        status = f"The episode is still running after step {last}."
    state = _clip((job.states or {}).get(job.uncertain_steps[-1], "") if job.uncertain_steps else "", 500)
    shown = "\n".join(shown_blocks.values()) if shown_blocks else "(none)"
    existing = "\n".join(existing_blocks.values()) if existing_blocks else "(none)"
    if (features or {}).get("judge_pool"):
        existing_header = ("Components retrieval ranks for this situation "
                           "(name [kind] utility/uses/failure-streak; best first). "
                           "If one of them already covers this situation, REVISE it "
                           "or answer NONE -- do not write a variant of it:")
    else:
        existing_header = ("Existing components for this task "
                           "(name [kind] utility/uses/failure-streak/blocked; "
                           "REVISE one to fix it):")
    blocked = ""
    if blocked_blocks:
        blocked = ("\nBlocked components (three consecutive failures; revise to retry):\n"
                   + "\n".join(blocked_blocks.values()) + "\n")
    user = JUDGE_TEMPLATE.format(
        env=job.env_name, task=job.task, goal=job.goal,
        trajectory=_trajectory(job, features=features), status=status,
        state=state or "(none)", shown=shown,
        existing_header=existing_header, existing=existing, blocked=blocked)
    system = review_system_prompt(features)
    if features and features.get("world_model"):
        system += ("\n\nS1 also sees, next to each legal action, the effect predicted by the "
                   "environment's world model (the trajectory shows [predicted: ...] for the executed action). "
                   "The world model is maintained separately from its checked predictions: do not "
                   "write a component that only says what an action does.")
    if features and features.get("progress"):
        system += ("\n\nUse the per-step reward column. Recommend only an action "
                   "that produced a positive reward delta or that advanced the "
                   "episode toward the final success. Never recommend an action "
                   "that produced no reward change, or the action on a FATAL step.")
    return system, user


def _common(job: ReviewJob, target_block: Optional[str], max_actions: int = 40) -> dict:
    actions = job.candidates[:max_actions]
    action_text = "; ".join(actions) if actions else "(none)"
    if len(job.candidates) > len(actions):
        action_text += f"; ... ({len(job.candidates) - len(actions)} more)"
    target = ""
    if target_block:
        target = ("\nCurrent version (your reply replaces it entirely):\n"
                  + target_block + "\n")
    step = job.uncertain_steps[-1] if job.uncertain_steps else 0
    return dict(step=step, observation=_clip(job.observation, 1200),
                inventory=_clip(job.inventory, 300) or "(empty)",
                actions=action_text, target=target)


def build_develop_prompt(kind: str, job: ReviewJob,
                         target_block: Optional[str] = None,
                         features: Optional[dict] = None) -> str:
    values = _common(job, target_block)
    template = {"procedure": PROCEDURE_TEMPLATE, "state": STATE_TEMPLATE}.get(
        kind, PROCEDURE_TEMPLATE)
    text = template.format(**values)
    features = features or {}
    if kind == "procedure" and features.get("positive_only"):
        text += ("\nThe body must contain positive guidance only. Never phrase advice "
                 "as \"do not X\", \"never X\" or \"avoid X\".")
    if kind == "procedure" and features.get("avoid"):
        text += ("\nIf an action or object must be suppressed, add an optional "
                 "`avoid: [pattern]` line to the frontmatter instead of writing a "
                 "negation in the body.")
    if kind == "state" and features.get("state_quality"):
        text += ("\nReturn a small dict when there is more than one useful fact. For a "
                 "goal that asks to craft/collect/gather/make a count of items, include "
                 "a numeric field for each required item and its current count. Never "
                 "return a constant or a value that is unknown for every step.")
    return text


WORLD_SYSTEM_PROMPT = """You are System-2 of a dual-system agent. System-1 (S1) chooses actions; you never act. You maintain the environment's world model: a Python program that predicts what each legal action would do right now. One world model serves every task of this environment; the goal tells which task an episode is. Before every decision the framework shows each prediction's effect text next to its action in S1's list of legal actions. After every step it checks the prediction for the executed action against what really happened. A wrong prediction misleads S1: it skips actions predicted to fail or end the episode, and takes actions predicted to succeed."""

WORLD_TEMPLATE = """Environment: {env}
Task: {task}
Goal of the episode: {goal}

{current}
{misses}{state_memory}
This episode's steps (step | action | reward | ends | next observation; [state: ...] is the state memory before the action, [predicted: ...] what the world model said):
{trajectory}

Write the complete world model:
```markdown
---
name: short-kebab-case-name
kind: world_model
when: the environment and situations this model covers
keywords: [k1, k2, k3]
---
## Rules
- <one line per rule the program encodes>
## Program
```python
def predict(goal, observation, inventory, history, actions, outcomes):
    return {{}}
```
```
Interface (pure Python; json, re and math are available without import; no other imports):
- goal, observation, inventory: the current texts.
- history: this episode's "action -> next observation" strings, oldest first.
- actions: the legal actions of this step (exact strings, often more than a hundred).
- outcomes: executed actions of this task, most recent first (this episode and earlier train episodes; every task's for a task never seen in training): a list of {{"action", "before", "after", "same", "reward", "ends", "inventory", "episode"}}. "before"/"after" are the observations before and after the action (whitespace-normalized, first 300 characters); "same" is whether the observation stayed identical; "reward" is the step reward; "ends" is whether the episode ended; "inventory" is the inventory after; "episode" is "this" or "earlier".
Return {{action: prediction}} for the actions you can predict, where a prediction is a dict:
- "effect": what the action does now, in plain words (at most 160 characters). This is the only text S1 reads.
- at least one checkable field; the framework checks every field you give:
  "same": True/False -- the next observation is identical to the current one (nothing visible changes);
  "ends": True/False -- the episode ends right after this action;
  "reward": "+", "0" or "-" -- the sign of this step's reward;
  "see": [phrases] -- each phrase appears in the next observation;
  "carry": [phrases] -- each phrase appears in the next inventory.
Every claim in "effect" must be backed by a checkable field. Omit an action rather than guess. The rules must hold for every task of the environment; when an action's effect depends on the task, decide it from the goal. Write rules that hold beyond the recorded steps: decide from the current observation and inventory what each action does, and use outcomes to learn and confirm those rules rather than only to look up a past result. Never return a dump of a state or of an outcome.
A generic example to adapt (not specific to any environment):
```python
def predict(goal, observation, inventory, history, actions, outcomes):
    now = " ".join(observation.split())[:300]
    last = {{}}                                  # most recent outcome of each action
    for o in outcomes:
        last.setdefault(o["action"], o)
    out = {{}}
    for a in actions:
        o = last.get(a)
        if o is not None and o["before"] == now:  # same action, same situation
            sign = "+" if o["reward"] > 0 else ("-" if o["reward"] < 0 else "0")
            text = "nothing changes" if o["same"] else "leads to: " + o["after"][:100]
            out[a] = {{"effect": text, "same": o["same"], "ends": o["ends"], "reward": sign}}
    return out
```
Reply NONE if no checkable rule can be written yet."""


def _world_trajectory(job: ReviewJob, max_steps: int = 40) -> str:
    rows = []
    for ev in job.trajectory[-max_steps:]:
        effect = ""
        if ev.get("effect"):
            mark = {True: "", False: " -- WRONG"}.get(ev.get("prediction_ok"), "")
            effect = f" [predicted: {_clip(ev.get('effect'), 80)}{mark}]"
        reward = ev.get("reward")
        state = ""
        for line in str(ev.get("state") or "").splitlines():
            if line.startswith("current: "):
                state = f" [state: {_clip(line[len('current: '):], 80)}]"
        rows.append(f"  {int(ev.get('step', 0))} |{state} {_clip(ev.get('action', ''), 100)}{effect} | "
                    f"{float(reward or 0):+g} | {'ends' if ev.get('done') else '-'} | "
                    f"{_clip(ev.get('feedback', ''), 160)}")
    if len(job.trajectory) > max_steps:
        rows.insert(0, f"  ... ({len(job.trajectory) - max_steps} earlier steps)")
    return "\n".join(rows) or "  (none)"


def render_miss(miss: dict) -> str:
    pred = dict(miss.get("predicted") or {})
    effect = pred.pop("effect", "")
    act = miss.get("actual") or {}
    lines = [f"- action: {miss.get('action', '')}" + (f" (step {miss['step']})" if "step" in miss else ""),
             f"  observation before: {_clip(miss.get('before', ''), 400)}"]
    if miss.get("inventory"):
        lines.append(f"  inventory before: {_clip(miss.get('inventory'), 150)}")
    if miss.get("state"):
        lines.append(f"  state before: {_clip(miss.get('state'), 160)}")
    lines.append(f"  predicted: \"{effect}\" {pred}")
    lines.append(f"  actually: same={act.get('same')}, ends={act.get('ends')}, "
                 f"reward {act.get('reward')}; next observation: {_clip(act.get('observation', ''), 300)}"
                 + (f"; inventory: {_clip(act.get('inventory'), 150)}" if act.get("inventory") else ""))
    lines.append(f"  wrong fields: {', '.join(miss.get('failed') or [])}")
    return "\n".join(lines)


def build_world_prompt(job: ReviewJob, target_block: Optional[str] = None,
                       features: Optional[dict] = None) -> Tuple[str, str]:
    """(system, user) for writing (target_block None) or revising a world model."""
    if target_block:
        current = ("Current world model (your reply replaces it entirely):\n" + target_block + "\n")
    else:
        current = "There is no world model for this environment yet.\n"
    misses = ""
    if job.misses:
        misses = ("Predictions in this episode that were wrong (predicted vs what "
                  "happened):\n" + "\n".join(render_miss(m) for m in job.misses) + "\n")
    state_memory = ""
    if job.state_memory:
        state_memory = ("State memory (the framework's graph of the states this task's state "
                        "program named, and the actions observed between them in this and earlier "
                        "episodes; state --action--> next state (times seen)):\n"
                        + job.state_memory + "\n")
    user = WORLD_TEMPLATE.format(env=job.env_name, task=job.task, goal=job.goal,
                                 current=current, misses=misses, state_memory=state_memory,
                                 trajectory=_world_trajectory(job))
    return WORLD_SYSTEM_PROMPT, user


def fix_prompt(problem: str) -> str:
    return FIX_TEMPLATE.format(problem=problem.strip())


__all__ = [
    "DECISION_LINE", "REVIEW_SYSTEM_PROMPT", "build_judge_prompt",
    "build_develop_prompt", "build_world_prompt", "fix_prompt", "judge_kinds",
    "parse_judge", "parse_skill_markdown", "render_miss", "review_system_prompt", "KINDS",
]
