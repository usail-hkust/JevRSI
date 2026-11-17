"""OpenAI-compatible chat client (works with any vLLM-served LLM/VLM).

Thin wrapper returning the assistant text plus token usage so the agent can
account for System-2 cost. Supports both single-shot (``chat``) and multi-turn
(``chat_messages``) conversations; the latter is used by the pure-LLM ReAct
baseline, which maintains a sliding window of messages.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List

# Inline thinking blocks some gateways emit inside `content` (Qwen-style).
# Parsers must see only the response, never the deliberation.
_THINK_BLOCK_RE = re.compile(r"<\|?think\|?>.*?<\|?/?think\|?>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<\|?think\|?>.*", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"<\|?/think\|?>", re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Drop inline think blocks; keep only the response part."""
    if "<" not in text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    # Unterminated opening tag (truncated stream): everything after is think.
    text = _THINK_OPEN_RE.sub("", text)
    # Closing tag only (the chat template opened the block in the prompt):
    # everything before it is think.
    text = _THINK_CLOSE_RE.split(text)[-1]
    return text.strip()


def _usage(resp) -> tuple:
    usage = getattr(resp, "usage", None)
    return (getattr(usage, "prompt_tokens", 0) or 0), (getattr(usage, "completion_tokens", 0) or 0)


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen2.5-7B-Instruct",
        api_key: str = "EMPTY",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        extra_body: dict | None = None,
        thinking_budget: int = 0,
        thinking_token_budget: int = 0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("openai package is required for the LLM backend") from exc
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        # None: omit the field (some gateway models accept only their default)
        self.temperature = None if temperature is None else float(temperature)
        self.max_tokens = max_tokens
        # e.g. {"chat_template_kwargs": {"enable_thinking": False}} to turn off
        # Qwen3.8's default thinking mode for faster, direct skill output.
        self.extra_body = extra_body or {}
        # Client-side budget forcing (> 0, thinking on): the first call is
        # capped at this many tokens; if it was cut off, a continuation with
        # ``continue_final_message`` closes the think block and answers.
        self.thinking_budget = max(0, int(thinking_budget or 0))
        # Server-side budget (vLLM's ``thinking_token_budget`` request field):
        # the engine forces the end-of-thinking string when the deliberation
        # reaches this many tokens, then generation continues into the answer.
        # Requires the vLLM server to run with --reasoning-parser/-config (see
        # scripts/serve_llm_vllm.sh); it takes precedence over the client-side
        # forcing above.
        self.thinking_token_budget = max(0, int(thinking_token_budget or 0))

    def _request_body(self, messages: List[Dict[str, str]], **more) -> dict:
        """Common request fields; temperature is left out when it is None."""
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                "extra_body": self._request_extra_body()}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(more)
        return body

    def _request_extra_body(self) -> dict | None:
        """The request body extras: configured extras + the server-side cap."""
        body = dict(self.extra_body or {})
        if self.thinking_token_budget:
            body["thinking_token_budget"] = self.thinking_token_budget
        return body or None

    def chat(self, system: str, user: str) -> ChatResult:
        """Single-shot system+user chat."""
        return self.chat_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    def chat_messages(self, messages: List[Dict[str, str]]) -> ChatResult:
        """Multi-turn chat over an OpenAI-style message list.

        Only the final ``content`` (the response) is returned -- reasoning /
        thinking is deliberately dropped, even when a gateway inlines it as
        ``<think>...</think>`` / ``<|think|>...`` blocks inside ``content``:
        action and skill parsing must operate on the response, never on the
        model's deliberation.
        """
        start = time.perf_counter()
        if self.thinking_budget and not self.thinking_token_budget:
            return self._chat_with_budget(messages, start)
        resp = self._client.chat.completions.create(**self._request_body(messages))
        latency = (time.perf_counter() - start) * 1000.0
        text = _strip_think(resp.choices[0].message.content or "")
        usage = getattr(resp, "usage", None)
        return ChatResult(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency,
        )

    def _chat_with_budget(self, messages: List[Dict[str, str]], start: float) -> ChatResult:
        """Thinking capped at ``thinking_budget`` tokens (client-side budget forcing).

        The first request may spend the budget; if it is cut off -- still
        thinking, or already answering -- the reply so far is continued as the
        assistant's own message (``continue_final_message``), with the think
        block closed first when it is still open.
        """
        first = self._client.chat.completions.create(**self._request_body(
            messages, max_tokens=self.thinking_budget))
        raw = first.choices[0].message.content or ""
        prompt, completion = _usage(first)
        if first.choices[0].finish_reason == "length":
            prefix = raw if _THINK_OPEN_RE.match(raw.lstrip()) else "<think>\n" + raw
            if not _THINK_CLOSE_RE.search(raw):
                prefix += "\n\nI must stop thinking and write the answer now.\n</think>\n\n"
            body = dict(self._request_extra_body() or {},
                        continue_final_message=True, add_generation_prompt=False)
            more = self._client.chat.completions.create(**self._request_body(
                list(messages) + [{"role": "assistant", "content": prefix}],
                extra_body=body))
            raw = prefix + (more.choices[0].message.content or "")
            p, c = _usage(more)
            prompt, completion = prompt + p, completion + c
        return ChatResult(text=_strip_think(raw), prompt_tokens=prompt, completion_tokens=completion,
                          latency_ms=(time.perf_counter() - start) * 1000.0)

    def close(self) -> None:
        return None
