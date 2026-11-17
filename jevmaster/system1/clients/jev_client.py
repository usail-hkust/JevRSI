"""HTTP client for a Jev-style decision model (System One ``/v1/systemone``).

Primary dialect (d3's native server, ``d3_server.py``):

    request : {"model": ..., "state": "...",
               "questions": {"action": {"type": "choice",
                                        "instructions": "...",
                                        "criteria": {"<option>": "<desc>", ...}}}}
    response: {"model", "answers": {"action": {"type": "choice", "choice": "<opt>",
                                               "probabilities": {opt: p, ...}}},
               "usage": {...}}

Other Jev-like backends answering the TypeSafe list dialect are still parsed
(defensive support for ``decisions``/``probs`` shapes), so any of
Jev-vllm / Kev / Clef / hosted Jev keeps working via ``base_url``.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import httpx

logger = logging.getLogger(__name__)

# Transient failures: wait and retry until the call succeeds. Other errors
# (auth / bad payload, e.g. 400/401/422) raise immediately -- retrying cannot
# fix them. 520-530 are the non-standard Cloudflare/gateway "origin
# overloaded" family (529 killed a 4-hour run on 2026-10-16 when it was not
# retried); 501 (not implemented) stays excluded as a permanent error.
RETRYABLE_STATUS = frozenset(
    {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 529, 530}
)


class JevClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8100",
        endpoint: str = "/v1/systemone",
        model: str = "d3",
        api_key: str = "EMPTY",
        timeout: float = 30.0,
        min_interval: float = 2.0,
        retry_delay: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
        self.model = model
        self.timeout = timeout
        # Pacing: at most one request every `min_interval` seconds, and on a
        # transient failure (rate limit / 5xx / network hiccup) wait
        # `retry_delay` seconds and retry until the call succeeds.
        self.min_interval = float(min_interval)
        self.retry_delay = float(retry_delay)
        self._last_request_at = 0.0
        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    def _throttle(self) -> None:
        """Sleep just enough to keep >= min_interval between request starts."""
        if self.min_interval <= 0:
            return
        wait = self._last_request_at + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _post_with_retry(self, payload: dict) -> httpx.Response:
        """POST the payload; on transient failure sleep and retry until success.

        Retries forever on 408/429/5xx and network/timeout errors (logging each
        attempt), so long experiments survive gateway rate limits. A server
        ``Retry-After`` hint larger than ``retry_delay`` is honored (capped at
        60s). Non-transient HTTP errors raise immediately.
        """
        while True:
            self._throttle()
            try:
                resp = self._client.post(self.endpoint, json=payload)
            except httpx.TransportError as exc:  # connection reset / timeout / ...
                logger.warning(
                    "Jev request failed (%s); retrying in %.0fs", exc, self.retry_delay
                )
                time.sleep(self.retry_delay)
                continue
            if resp.status_code in RETRYABLE_STATUS:
                delay = self.retry_delay
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, min(float(retry_after), 60.0))
                    except ValueError:
                        pass
                logger.warning(
                    "Jev request failed (HTTP %d); retrying in %.0fs",
                    resp.status_code, delay,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp

    def choice(
        self, state: str, options: List[str], qid: str = "action",
        instructions: str = "Choose the single best next action.",
        criteria: Dict[str, str] = None,
    ) -> Tuple[str, float, Dict[str, float], dict]:
        """Ask a single ``choice`` question and return (value, confidence, probs, raw).

        ``options`` are the option KEYS (returned by the model); ``criteria``
        optionally maps key -> text description shown to the model (Jev-paper
        convention: neutral keys like ``c0``/``c1`` with the action text as
        the description; defaults to ``{key: key}``).
        """
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                qid: {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria if criteria is not None else {opt: opt for opt in options},
                }
            },
        }
        resp = self._post_with_retry(payload)
        data = resp.json()

        entry = self._answer_entry(data, qid)
        value = entry.get("choice", entry.get("answer", entry.get("value")))
        probs = self._parse_probs(entry, options)
        confidence = float(entry.get("p", 0.0) or 0.0)
        if not confidence and value is not None:
            confidence = probs.get(str(value), 0.0)
        if value is None and probs:
            value = max(probs, key=probs.get)
            confidence = probs[value]
        return value, confidence, probs, data

    @staticmethod
    def _answer_entry(data: dict, qid: str) -> dict:
        """Locate the answer object, d3-native first, TypeSafe-list fallback."""
        if not isinstance(data, dict):
            return {}
        answers = data.get("answers")
        if isinstance(answers, dict) and qid in answers:
            entry = answers[qid]
            return entry if isinstance(entry, dict) else {}
        decisions = data.get("decisions", [])
        for d in decisions:
            if isinstance(d, dict) and d.get("id") == qid:
                return d
        if isinstance(decisions, list) and decisions:
            d = decisions[0]
            return d if isinstance(d, dict) else {}
        # d3 may answer with a bare dict keyed by question id at top level.
        top = data.get(qid)
        return top if isinstance(top, dict) else {}

    @staticmethod
    def _parse_probs(decision: dict, options: List[str]) -> Dict[str, float]:
        # d3 native: {"probabilities": {option: share}}
        probs = decision.get("probabilities", decision.get("probs"))
        if isinstance(probs, dict):
            return {str(k): float(v) for k, v in probs.items()}
        if isinstance(probs, list) and len(probs) == len(options):
            return {opt: float(p) for opt, p in zip(options, probs)}
        # TypeSafe variants: {"options": [{"value"/"option": .., "p": ..}]}
        opt_list = decision.get("options")
        if isinstance(opt_list, list):
            out: Dict[str, float] = {}
            for item in opt_list:
                if isinstance(item, dict):
                    key = item.get("value", item.get("option"))
                    if key is not None and "p" in item:
                        out[str(key)] = float(item["p"])
            if out:
                return out
        # Fallback: winning value + its p
        value = decision.get("choice", decision.get("answer", decision.get("value")))
        if value is not None and "p" in decision:
            return {str(value): float(decision["p"])}
        return {}

    def close(self) -> None:
        self._client.close()
