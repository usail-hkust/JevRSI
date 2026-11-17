"""MiniGrid / BabyAI adapter.

Two episode-set shapes (``env.params.items_file``):

- **split manifest** (current protocol): records with ``task`` (a BabyAI env
  id) and ``variation`` (the reset seed), frozen under
  ``data/minigrid/splits/`` by ``scripts/make_minigrid_split.py``. The
  protocol uses ``BabyAI-MixedTrainLocal-v0`` for train and ID (disjoint
  seeds) and ``BabyAI-MixedTestLocal-v0`` for OOD.
- **AgentGym AgentEval list** (legacy): the 90-item BabyAI test set over the
  fork (BartekCupial/Minigrid, the one BALROG uses). AgentGym's addressing:
  ``level = all_levels[data_idx % 40 + 1]``, ``seed = data_idx // 40`` -- the
  40-level table is copied verbatim as ``AGENTGYM_LEVELS`` (index 0 =
  all_levels[1]). ``task`` is the level id and ``variation`` is the reset
  seed, so episode ids and skills key on the level.

**Mixed-level determinism.** ``Level_MixedTrainLocal`` / ``Level_MixedTestLocal``
pick their action kind (goto / pickup / open / putnext / seq) in ``__init__``
with an *unseeded* RNG, so the same reset seed otherwise produces different
tasks in different processes. ``make_mixed_env`` pre-seeds that construction
RNG with the episode seed; the kind and the map are then both deterministic
while train and ID still see the full MixedTrain task distribution. The
adapter caches one environment per (env id, seed).

Observations: the fork's egocentric textual descriptions (what the agent
carries, walls in each direction, visible objects with relative step
distances) with the "You see " prefixes stripped and lines joined, exactly
BALROG's ``_form_prompt``. The mission goes to ``Observation.goal``.
Actions: BALROG's six text actions; the tuple index is the minigrid action id
(left=0 ... toggle=5; the "done" action is not exposed). Score: 1.0 once any
positive reward is seen (mission completed), else 0.0 (AgentEval's BabyAI
reward is the win flag).

The fork ships ``BabyAI-GoToObjS6-v1`` (a registry-parameter fix of the
original ``-v0``); ``-v0`` is aliased to it at init so the AgentGym level
table resolves as-is. Requires the fork
(``git clone https://github.com/BartekCupial/Minigrid`` +
``pip install -e ./Minigrid``; pulls gymnasium). Imported lazily so importing
this package never requires it.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jevmaster.envs.agenteval import ITEM_FILES, load_item_indices
from jevmaster.envs.base import EnvAdapter
from jevmaster.envs.registry import register_env
from jevmaster.envs.splits import SPLITS, load_items
from jevmaster.types import Candidate, Observation, StepResult

logger = logging.getLogger(__name__)

# The two mixed BabyAI envs used by the split protocol. Their action kind is
# chosen in __init__ (see make_mixed_env).
MIXED_ENVS: Dict[str, str] = {
    "BabyAI-MixedTrainLocal-v0": "Level_MixedTrainLocal",
    "BabyAI-MixedTestLocal-v0": "Level_MixedTestLocal",
}
MIXED_KINDS: Tuple[str, ...] = (
    "goto", "pickup", "open", "putnext", "pick up seq go to",
)


def make_mixed_env(env_id: str, seed: int):
    """A MixedTrain/MixedTest env whose construction RNG is seeded.

    ``Level_MixedTrainLocal.__init__`` picks its action kind before the gym
    reset seed exists; seeding ``_np_random`` first makes that choice a pure
    function of the episode seed. ``reset(seed=seed)`` then regenerates the
    map/mission deterministically.
    """
    import numpy as np
    from minigrid.envs.babyai.mixed_seq_levels import (
        Level_MixedTestLocal,
        Level_MixedTrainLocal,
    )

    classes = {
        "BabyAI-MixedTrainLocal-v0": Level_MixedTrainLocal,
        "BabyAI-MixedTestLocal-v0": Level_MixedTestLocal,
    }
    if env_id not in classes:
        raise ValueError(f"minigrid: not a mixed BabyAI env: {env_id!r}")
    cls = classes[env_id]
    env = cls.__new__(cls)
    env._np_random = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    cls.__init__(env)
    return env


def load_episode_items(items_file: str, split: Optional[str] = None):
    """Dispatch to ``("manifest", records)`` or ``("agentgym", data_idx list)``.

    A manifest record has ``task`` and ``variation``; an AgentEval record is
    ``{"item_id": "babyai_<data_idx>"}``. ``split`` (train|id|ood) filters a
    combined manifest and is rejected for the legacy list. Pure Python, so the
    dispatch is testable without MiniGrid.
    """
    path = Path(items_file)
    if not path.exists():
        raise FileNotFoundError(
            f"MiniGrid items file not found: {path} (run from the repo root "
            "or pass env.params.items_file)"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    records = data.get("items") if isinstance(data, dict) else data
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty item array")
    is_manifest = all(isinstance(r, dict) and "task" in r and "variation" in r
                      for r in records)
    if is_manifest:
        if split is not None and split not in SPLITS:
            raise ValueError(
                f"minigrid: unknown split {split!r} (train | id | ood)")
        items = load_items(path, split=split)
        for i, record in enumerate(items):
            if not str(record.get("task") or "").strip():
                raise ValueError(f"{path}:{i}: missing task")
            if not isinstance(record.get("variation"), int):
                raise ValueError(f"{path}:{i}: variation must be an int")
        return "manifest", items
    if split is not None:
        raise ValueError(
            "minigrid: split=... selects a frozen manifest; the legacy "
            "AgentEval lists have no split")
    return "agentgym", load_item_indices(str(path))

# BALROG's BABYAI_ACTION_SPACE: the tuple index IS the minigrid action id.
ACTIONS = ("turn left", "turn right", "go forward", "pick up", "drop", "toggle")

# AgentGym's babyai ``all_levels`` table (agentenv-babyai environment.py),
# verbatim order: index i here = all_levels[i + 1], i.e. level =
# AGENTGYM_LEVELS[data_idx % 40], seed = data_idx // 40.
AGENTGYM_LEVELS: Tuple[str, ...] = (
    "BabyAI-GoToRedBallGrey-v0",       # 1
    "BabyAI-GoToRedBall-v0",           # 2
    "BabyAI-GoToRedBallNoDists-v0",    # 3
    "BabyAI-GoToObjS6-v0",             # 4
    "BabyAI-GoToLocalS8N7-v0",         # 5
    "BabyAI-GoToObjMazeS7-v0",         # 6
    "BabyAI-GoToImpUnlock-v0",         # 7
    "BabyAI-GoToSeqS5R2-v0",           # 8
    "BabyAI-GoToRedBlueBall-v0",       # 9
    "BabyAI-GoToDoor-v0",              # 10
    "BabyAI-GoToObjDoor-v0",           # 11
    "BabyAI-Open-v0",                  # 12
    "BabyAI-OpenRedDoor-v0",           # 13
    "BabyAI-OpenDoorLoc-v0",           # 14
    "BabyAI-OpenRedBlueDoorsDebug-v0",  # 15
    "BabyAI-OpenDoorsOrderN4Debug-v0",  # 16
    "BabyAI-Pickup-v0",                # 17
    "BabyAI-UnblockPickup-v0",         # 18
    "BabyAI-PickupLoc-v0",             # 19
    "BabyAI-PickupDistDebug-v0",       # 20
    "BabyAI-PickupAbove-v0",           # 21
    "BabyAI-PutNextLocalS6N4-v0",      # 22
    "BabyAI-PutNextS7N4Carrying-v0",   # 23
    "BabyAI-Unlock-v0",                # 24
    "BabyAI-UnlockLocalDist-v0",       # 25
    "BabyAI-KeyInBox-v0",              # 26
    "BabyAI-UnlockPickupDist-v0",      # 27
    "BabyAI-BlockedUnlockPickup-v0",   # 28
    "BabyAI-UnlockToUnlock-v0",        # 29
    "BabyAI-ActionObjDoor-v0",         # 30
    "BabyAI-FindObjS7-v0",             # 31
    "BabyAI-KeyCorridorS6R3-v0",       # 32
    "BabyAI-OneRoomS20-v0",            # 33
    "BabyAI-MoveTwoAcrossS8N9-v0",     # 34
    "BabyAI-SynthS5R2-v0",             # 35
    "BabyAI-SynthLoc-v0",              # 36
    "BabyAI-SynthSeq-v0",              # 37
    "BabyAI-MiniBossLevel-v0",         # 38
    "BabyAI-BossLevel-v0",             # 39
    "BabyAI-BossLevelNoUnlock-v0",     # 40
)

NOTES = (
    "Objects are described relative to your position and facing direction "
    "(e.g. 'a red ball 2 steps right and 1 step forward'): 'forward' is the "
    "direction you currently face. 'go forward' moves one cell ahead, "
    "'turn left'/'turn right' rotate you in place, 'pick up'/'drop'/'toggle' "
    "act on the cell in front of you. You only see part of the world."
)


def item_to_level_seed(data_idx: int) -> Tuple[str, int]:
    """AgentGym's babyai addressing: (level id, reset seed) for a data_idx."""
    return AGENTGYM_LEVELS[data_idx % len(AGENTGYM_LEVELS)], data_idx // len(AGENTGYM_LEVELS)


def _data_idx(level: str, seed: int) -> int:
    """Inverse of :func:`item_to_level_seed` (for episode extras)."""
    return AGENTGYM_LEVELS.index(level) + len(AGENTGYM_LEVELS) * seed







class MiniGridEnvAdapter(EnvAdapter):
    name = "minigrid"

    def __init__(
        self,
        items_file: str = ITEM_FILES["minigrid"],
        split: str = None,
        **_,
    ):
        try:
            import gymnasium
            import minigrid  # noqa: F401  (BALROG BabyAI-Text fork)
        except ImportError as exc:
            raise ImportError(
                "minigrid is not installed. The BALROG BabyAI-Text fork is "
                "required: git clone https://github.com/BartekCupial/Minigrid "
                "&& pip install -e ./Minigrid (pulls gymnasium)."
            ) from exc

        if "BabyAI-MixedTrainLocal-v0" not in gymnasium.envs.registry:
            try:
                minigrid.register_minigrid_envs()  # the fork does not auto-register
            except Exception as exc:  # pragma: no cover - PyPI minigrid layout differs
                raise ImportError(
                    "minigrid: the installed package is not the BALROG BabyAI-Text "
                    "fork. Install it with: git clone "
                    "https://github.com/BartekCupial/Minigrid && pip install -e ./Minigrid"
                ) from exc
            if "BabyAI-MixedTrainLocal-v0" not in gymnasium.envs.registry:
                raise ImportError(
                    "minigrid: the installed package is not the BALROG BabyAI-Text "
                    "fork (BabyAI-MixedTrainLocal-v0 is missing). Install it with: "
                    "git clone https://github.com/BartekCupial/Minigrid && "
                    "pip install -e ./Minigrid"
                )
        # The fork ships GoToObjS6 as -v1 (a registry-parameter fix); alias
        # AgentGym's -v0 name to it so the level table resolves as-is.
        if "BabyAI-GoToObjS6-v0" not in gymnasium.envs.registry:
            gymnasium.register(
                id="BabyAI-GoToObjS6-v0",
                entry_point="minigrid.envs.babyai:GoToObj",
                kwargs={"room_size": 6},
            )

        self._items: List[int] = []
        self._manifest_items: Optional[List[Dict[str, Any]]] = None
        mode, payload = load_episode_items(items_file, split)
        if mode == "manifest":
            self._manifest_items = payload
            missing = sorted({
                str(record["task"]) for record in payload
                if str(record["task"]) not in gymnasium.envs.registry
            })
            if missing:
                raise ValueError(
                    f"minigrid: env ids not registered in the installed fork: "
                    f"{missing}"
                )
        else:
            self._items = payload
            missing = sorted(
                {level for level, _ in map(item_to_level_seed, self._items)}
                - set(gymnasium.envs.registry)
            )
            if missing:
                raise ValueError(
                    f"minigrid: levels not registered in the installed fork: {missing}"
                )

        self._envs: Dict[Any, object] = {}
        self._env = None
        self._raw = None
        self._task = ""
        self._mission = ""
        self._seed = None
        self._descriptions: List[str] = []
        self._progression = 0.0

    def task_variations(self) -> List[Tuple[str, int]]:
        if self._manifest_items is not None:
            return [(str(record["task"]), int(record["variation"]))
                    for record in self._manifest_items]
        return [item_to_level_seed(d) for d in self._items]

    def _get_env(self, task: str, seed: int):
        """Lazily create (and cache) the env for one episode.

        Mixed BabyAI envs choose their task kind during construction, so they
        are built by ``make_mixed_env`` with that RNG pre-seeded by the episode
        seed. Everything else visible in an episode comes from the seeded
        ``reset`` below.
        """
        if task in MIXED_ENVS:
            key = (task, int(seed))
            if key not in self._envs:
                self._envs[key] = make_mixed_env(task, int(seed))
            return self._envs[key]
        if task in self._envs:
            return self._envs[task]
        import gymnasium

        env = gymnasium.make(task, disable_env_checker=True)
        self._envs[task] = env
        return env

    def reset(self, task: str, variation: int) -> Observation:
        env = self._get_env(task, variation)
        obs, info = env.reset(seed=int(variation))
        self._env = env
        self._raw = env.unwrapped
        self._task = task
        self._seed = int(variation)
        self._mission = str(obs.get("mission", "") or self._raw.mission or "")
        self._descriptions = list(info.get("descriptions") or [])
        self._progression = 0.0
        return self._observe()

    def _current_action_kind(self) -> Optional[str]:
        kinds = getattr(self._raw, "action_kinds", None) if self._raw else None
        return str(kinds[0]) if kinds else None

    def _observe(self) -> Observation:
        # BALROG BabyAITextCleanLangWrapper._form_prompt: strip the
        # "You see " prefixes and join the description lines.
        text = "\n".join(
            d.replace("You see ", "") for d in self._descriptions if d
        )
        carrying = getattr(self._raw, "carrying", None) if self._raw else None
        inventory = f"{carrying.color} {carrying.type}" if carrying else ""
        extra: Dict[str, Any] = {"seed": self._seed}
        if self._task in AGENTGYM_LEVELS:
            extra["data_idx"] = _data_idx(self._task, self._seed)
        action_kind = self._current_action_kind()
        if action_kind:
            extra["action_kind"] = action_kind
        return Observation(
            task=self._task,
            goal=self._mission,
            text=text,
            inventory=inventory,
            step=int(getattr(self._raw, "step_count", 0) or 0),
            notes=NOTES,
            extra=extra,
        )

    def candidates(self) -> List[Candidate]:
        return [Candidate(a) for a in ACTIONS]

    def step(self, action: str) -> StepResult:
        idx = ACTIONS.index(action)  # ValueError on illegal strings; agents only pass candidates
        obs, reward, terminated, truncated, info = self._env.step(idx)
        self._descriptions = list(info.get("descriptions") or self._descriptions)
        if reward > 0:
            self._progression = 1.0
        return StepResult(
            observation=self._observe(),
            reward=float(reward),
            done=bool(terminated or truncated),
            info={"terminated": bool(terminated), "truncated": bool(truncated)},
        )

    def score(self) -> float:
        # 1.0 once a positive reward was seen (mission completed), else 0.0 —
        # AgentEval's BabyAI reward is the win flag.
        return self._progression

    def close(self) -> None:
        for env in self._envs.values():
            try:
                env.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self._envs.clear()
        self._env = None
        self._raw = None


@register_env("minigrid")
def _make_minigrid(**kwargs) -> MiniGridEnvAdapter:
    return MiniGridEnvAdapter(**kwargs)
