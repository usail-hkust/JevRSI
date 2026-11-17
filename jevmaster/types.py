"""Core data types shared across the dual-system agent.

These are intentionally plain dataclasses with no dependency on any specific
environment, model backend, or serving framework, so that new envs / Jev-like
models / LLM backends can be added without touching the core contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    """Normalized observation produced by every EnvAdapter.

    Concrete adapters map their native observation into these fields so the
    rest of the system never needs to know which environment it is running.
    """

    task: str                      # task name / id (e.g. "boil")
    goal: str                      # natural-language goal / instruction
    text: str                      # main textual observation of the scene
    inventory: str = ""            # agent inventory, if the env exposes it
    step: int = 0                  # step index within the episode
    notes: str = ""                # env-specific caution/hint shown to the agent
    extra: Dict[str, Any] = field(default_factory=dict)  # env-specific fields

    def as_context(self) -> str:
        """Render the observation as a compact text block for a model prompt."""
        parts = [f"Task: {self.task}", f"Step: {self.step}", f"Goal: {self.goal}"]
        if self.inventory:
            parts.append(f"Inventory: {self.inventory}")
        parts.append(f"Observation: {self.text}")
        if self.notes:
            parts.append(f"Note: {self.notes}")
        return "\n".join(parts)


@dataclass
class Candidate:
    """A single legal action the agent may take this step."""

    action: str                    # the exact action string passed to env.step
    label: Optional[str] = None    # optional human-friendly label (defaults to action)
    template: Optional[str] = None  # action-type label (e.g. "focus on OBJ"); enables two-stage choice
    description: Optional[str] = None  # what this action would do here (its effect), shown per option

    def __post_init__(self) -> None:
        if self.label is None:
            self.label = self.action


@dataclass
class S1Decision:
    """Result of a System-1 action selection."""

    action: str                        # chosen action string
    confidence: float                  # top-choice probability in [0, 1]
    probs: Dict[str, float] = field(default_factory=dict)  # per-candidate prob
    latency_ms: float = 0.0
    tokens: int = 0                    # total tokens for the call (in+out)
    tokens_in: int = 0                 # input/prompt tokens (when the backend reports usage)
    tokens_out: int = 0                # output/completion tokens
    backend: str = ""                  # which S1 backend produced this
    # Two-stage choice (template -> object). `confidence` is the weakest
    # stage (min(p_template, p_action)): the trigger fires when System-1 is
    # uncertain at either stage. Both per-stage probabilities are kept
    # separately for analysis.
    two_stage: bool = False
    template: Optional[str] = None              # chosen action-type label
    p_template: Optional[float] = None          # stage-1 probability of that template
    template_probs: Dict[str, float] = field(default_factory=dict)  # stage-1 probs
    p_action: Optional[float] = None            # stage-2 conditional prob of the action




@dataclass
class StepResult:
    """Outcome of applying an action to the environment."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    # Set by adapters that expose an irreversible/fatal action outcome
    # Some environments use score/reward -100 for a fatal outcome.
    fatal: bool = False


@dataclass
class ReviewJob:
    """One System-2 review: uncertain System-1 decisions plus what followed.

    Built inside the System-2 process from the step events System-1 emits:
    the episode trajectory up to ``K`` steps after the (last) uncertain step,
    or up to the episode end, so System-2 can judge whether the decisions and
    the three skill channels achieved what was expected.
    """

    job_id: str
    env_name: str
    episode_id: str
    task: str
    goal: str
    trajectory: List[Dict[str, Any]]     # step events (step, action, confidence, feedback, reward, skills)
    uncertain_steps: List[int]           # steps under review (System-1 confidence < threshold)
    observation: str = ""                # state at the last uncertain step
    inventory: str = ""
    candidates: List[str] = field(default_factory=list)  # legal actions at that step
    outcome: Optional[Dict[str, Any]] = None  # episode_end event once the episode finished
    shown_skills: List[str] = field(default_factory=list)  # procedure names shown at the uncertain steps
    # State memory graph summary at the uncertain steps.
    states: Dict[int, str] = field(default_factory=dict)     # step -> state summary
    # Episode start, for replaying a state program in admission.
    start: Dict[str, Any] = field(default_factory=dict)
    state_program: Optional[str] = None
    state_errors: List[List[Any]] = field(default_factory=list)   # [step, message]
    # M5.3 world model maintenance (System-2 writes or revises without a judge turn)
    kind_hint: Optional[str] = None      # "world_model" for a world-model job
    target: Optional[str] = None         # world model to revise (None: write one)
    misses: List[Dict[str, Any]] = field(default_factory=list)   # wrong checked predictions
    state_memory: str = ""               # the episode's state graph: states and the actions between them


@dataclass
class ReviewResult:
    """System-2's answer to a :class:`ReviewJob` (judge turn + optional develop turn)."""

    decision: str = "NONE"               # NONE | NEW | REVISE
    target: Optional[str] = None         # skill name for REVISE
    reason: str = ""
    skill_markdown: Optional[str] = None  # develop-turn output (None: declined / nothing written)
    kind: str = "procedure"              # skill kind (jevmaster.skills.skill)
    cooldown: bool = False               # REVISE target was revised too recently: not developed
    chats: List[Dict[str, Any]] = field(default_factory=list)  # [{turn, messages, reply, tokens}]
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class StepRecord:
    """One row of the supervision/cost log (Resource.md Section F).

    Written as JSONL. Fields capture everything needed later to build routing
    supervision and to analyze reasoning amortization.
    """

    episode_id: str
    env_name: str
    task: str
    step: int
    goal: str
    observation: str
    inventory: str
    candidates: List[str]
    s1_action: str
    s1_confidence: float
    s1_probs: Dict[str, float]
    s1_latency_ms: float
    s1_tokens: int
    s2_triggered: bool
    s2_job_id: Optional[str]
    reward: float
    cumulative_reward: float
    done: bool
    retrieved_skills: List[str]
    skill_version: int                 # SkillStore revision at decision time
    s1_backend: str
    mode: str = "dual"                 # dual | jev | llm_agent
    exp_id: str = ""                   # experiment id this row belongs to
    next_observation: str = ""         # environment state AFTER the action
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
