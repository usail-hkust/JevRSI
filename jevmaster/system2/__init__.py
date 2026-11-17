"""System-2: the asynchronous skill reviewer (LLM-based).

System-2 runs in its own process (:mod:`jevmaster.system2.service`) and
maintains the three skill kinds.  When System-1 is uncertain it judges the
decision against what followed and develops or revises at most one component.

This module also exposes :func:`build_openai_client`, the shared constructor
for the OpenAI-compatible chat client (local vLLM).
"""
from typing import Optional

from jevmaster.system2.base import System2
from jevmaster.system2.llm import LLMSystem2
from jevmaster.system2.clients.local_stub import StubSystem2
from jevmaster.system2.service import S2Process, S2Service

__all__ = ["System2", "LLMSystem2", "StubSystem2", "S2Process", "S2Service",
           "build_system2", "build_openai_client"]


def _section(config):
    return config.section("system2") if hasattr(config, "section") else config


def build_openai_client(config, develop: bool = False, overrides: Optional[dict] = None):
    """Build the shared OpenAI-compatible chat client from ``system2.llm``."""
    from jevmaster.system2.clients.openai_client import OpenAIChatClient

    section = config.section("system2")
    llm_cfg = dict(section.get("llm", {}) or {})
    if develop:
        llm_cfg.update(llm_cfg.get("develop") or {})
    llm_cfg.update(overrides or {})
    return OpenAIChatClient(
        base_url=llm_cfg.get("base_url", "http://localhost:8000/v1"),
        model=llm_cfg.get("model", "Qwen3.8-27B"),
        api_key=llm_cfg.get("api_key", "EMPTY"),
        temperature=float(llm_cfg.get("temperature", 0.3)),
        max_tokens=int(llm_cfg.get("max_tokens", 8192)),
        timeout=float(llm_cfg.get("timeout", 60)),
        extra_body=llm_cfg.get("extra_body") or {},
        thinking_budget=int(llm_cfg.get("thinking_budget", 0) or 0),
        thinking_token_budget=int(llm_cfg.get("thinking_token_budget", 0) or 0))


def build_system2(config):
    """Factory for the configured System-2 backend (None when backend: none)."""
    section = _section(config)
    backend = (section.get("backend") or "stub").lower()
    if backend == "none":
        return None
    if backend == "llm":
        llm_cfg = section.get("llm", {}) or {}
        develop = build_openai_client(config, develop=True) if llm_cfg.get("develop") else None
        return LLMSystem2(build_openai_client(config), develop_client=develop)
    if backend == "stub":
        return StubSystem2(**(section.get("stub", {}) or {}))
    raise ValueError(f"Unknown system2 backend: {backend}")
