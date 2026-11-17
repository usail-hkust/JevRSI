"""Configuration loading and typed access.

A config is a deep-merge of (1) ``configs/base.yaml``, (2) an optional
environment-specific override file, and (3) CLI overrides. The merged mapping is
wrapped in :class:`Config`, which offers dotted-path access plus a few helpers
for the sub-sections used across the codebase.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@dataclass
class Config:
    """Thin wrapper around a merged configuration mapping."""

    data: Dict[str, Any]

    def get(self, path: str, default: Any = None) -> Any:
        """Fetch a value via a dotted path, e.g. ``config.get("system1.backend")``."""
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, path: str) -> Dict[str, Any]:
        """Return a sub-mapping (empty dict if missing)."""
        value = self.get(path, {})
        return value if isinstance(value, dict) else {}

    def set(self, path: str, value: Any) -> None:
        """Set a value via a dotted path, creating intermediate dicts."""
        node = self.data
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def load_config(
    path: str | Path,
    base_path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load and merge configuration.

    Parameters
    ----------
    path:
        Primary config file (e.g. ``configs/alfworld.yaml``).
    base_path:
        Optional base config to merge under ``path``. If omitted, a sibling
        ``base.yaml`` next to ``path`` is used when present.
    overrides:
        Flat mapping of dotted-path -> value applied last (e.g. from the CLI).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        primary = yaml.safe_load(f) or {}

    if base_path is None:
        candidate = path.parent / "base.yaml"
        base_path = candidate if candidate.exists() and candidate != path else None

    merged: Dict[str, Any] = {}
    if base_path is not None:
        with open(base_path, "r", encoding="utf-8") as f:
            merged = yaml.safe_load(f) or {}
    merged = _deep_merge(merged, primary)

    config = Config(merged)
    for dotted, value in (overrides or {}).items():
        if value is not None:
            config.set(dotted, value)
    return config
