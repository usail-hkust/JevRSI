"""A tiny name -> factory registry for environment adapters.

New datasets are added by writing an adapter and decorating its factory with
``@register_env("name")`` -- no change to the agent core.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from jevmaster.envs.base import EnvAdapter

_REGISTRY: Dict[str, Callable[..., EnvAdapter]] = {}


def register_env(name: str) -> Callable[[Callable[..., EnvAdapter]], Callable[..., EnvAdapter]]:
    def decorator(factory: Callable[..., EnvAdapter]) -> Callable[..., EnvAdapter]:
        _REGISTRY[name] = factory
        return factory

    return decorator


def make_env(name: str, **kwargs) -> EnvAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown env '{name}'. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def available_envs() -> List[str]:
    return sorted(_REGISTRY)
