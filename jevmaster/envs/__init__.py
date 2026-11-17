"""Environment adapters that normalize every benchmark to one interface."""

from jevmaster.envs.base import EnvAdapter
from jevmaster.envs.registry import register_env, make_env, available_envs

# Import adapters for their registration side effects. Heavy optional
# dependencies (alfworld, minigrid) are imported
# lazily inside each factory, so importing this package never requires them
# to be installed.
from jevmaster.envs import mock_env  # noqa: F401
from jevmaster.envs import alfworld_env  # noqa: F401
from jevmaster.envs import minigrid_env  # noqa: F401
from jevmaster.envs import textcraft_synth_env  # noqa: F401

__all__ = ["EnvAdapter", "register_env", "make_env", "available_envs"]
