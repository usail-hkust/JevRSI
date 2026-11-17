"""System-1: the fast, always-in-control decision maker (Jev-style)."""

from jevmaster.system1.base import System1
from jevmaster.system1.jev import JevSystem1
from jevmaster.system1.clients.local_stub import StubSystem1

__all__ = ["System1", "JevSystem1", "StubSystem1", "build_system1"]


def _section(config):
    """Accept a full Config (preferred) or the system1 section itself."""
    return config.section("system1") if hasattr(config, "section") else config


def build_system1(config) -> System1:
    """Factory: construct the configured System-1 backend.

    ``config`` is the full :class:`Config` (or the ``system1`` section).
    Backends: ``jev`` (the choice model), ``llm`` (the Qwen chat service of
    ``system2.llm``, ``system1.llm`` overriding any of its keys -- the aligned
    llm_agent baseline: the same loop, a different decider; ``run.mode: llm``
    uses the pure prompt: interaction context + legal actions only) and ``stub``
    (offline tests).
    """
    section = _section(config)
    backend = (section.get("backend") or "stub").lower()
    mode = str(config.get("run.mode") or "").lower() if hasattr(config, "get") else ""
    pure = mode == "llm"
    if backend == "llm":
        from jevmaster.system1.llm_s1 import LLMSystem1
        from jevmaster.system2 import build_openai_client

        client_cfg = (section.get("llm", {}) or {})
        if str(client_cfg.get("backend") or "").lower() == "stub":
            from jevmaster.system2.clients.local_chat_stub import StubChatClient

            return LLMSystem1(StubChatClient(**(client_cfg.get("stub", {}) or {})), pure=pure)
        return LLMSystem1(build_openai_client(config, overrides=client_cfg), pure=pure)
    if backend == "jev":
        from jevmaster.system1.clients.jev_client import JevClient

        jev_cfg = section.get("jev", {}) or {}
        client = JevClient(
                base_url=jev_cfg.get("base_url", "http://localhost:8100"),
                endpoint=jev_cfg.get("endpoint", "/v1/systemone"),
                model=jev_cfg.get("model", "d3"),
                api_key=jev_cfg.get("api_key", "EMPTY"),
                timeout=float(jev_cfg.get("timeout", 30)),
                # Pacing exists for the gateway's rate limit; a local server
                # answers back-to-back.
                min_interval=float(jev_cfg.get("local_min_interval", jev_cfg.get("min_interval", 2.0))),
                retry_delay=float(jev_cfg.get("retry_delay", 5.0)),
            )
        return JevSystem1(
            client,
            max_candidates=int(jev_cfg.get("max_candidates", 255)),
            two_stage=bool(jev_cfg.get("two_stage", True)),
        )
    if backend == "stub":
        return StubSystem1(**(section.get("stub", {}) or {}))
    raise ValueError(f"Unknown system1 backend: {backend}")
