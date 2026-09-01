"""Telemetry capture for bake-off eval runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolCallMetric:
    name: str
    args: dict[str, Any]
    latency_ms: float | None = None
    result_preview: str | None = None


@dataclass
class RunTelemetry:
    task_id: str
    mode: str
    thread_id: str
    model: str
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    latency_ms: float | None = None
    tool_calls: list[ToolCallMetric] = field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    interrupted: bool = False
    approval_requested: bool = False
    pass_fail: str | None = None
    score_details: dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    error: str | None = None

    def finish(self) -> None:
        self.ended_at = time.time()
        self.latency_ms = round((self.ended_at - self.started_at) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Rough mid-tier pricing for cost estimates (USD per 1M tokens)
_MODEL_PRICES = {
    "anthropic:claude-sonnet-4-6": (3.0, 15.0),
    "openai:gpt-4o": (2.5, 10.0),
    "openai:gpt-4o-mini": (0.15, 0.6),
}


def estimate_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    prices = _MODEL_PRICES.get(model)
    if not prices:
        # default mid-tier guess
        prices = (3.0, 15.0)
    inp, out = prices
    return round((input_tokens * inp + output_tokens * out) / 1_000_000, 6)


def extract_token_usage(messages: list[Any]) -> tuple[int | None, int | None, int | None]:
    """Best-effort token usage extraction from LangChain message metadata."""
    in_tok = 0
    out_tok = 0
    found = False
    for msg in messages:
        meta = getattr(msg, "usage_metadata", None) or {}
        if not meta:
            resp = getattr(msg, "response_metadata", None) or {}
            meta = resp.get("usage") or resp.get("token_usage") or {}
        if not meta:
            continue
        found = True
        in_tok += int(meta.get("input_tokens") or meta.get("prompt_tokens") or 0)
        out_tok += int(meta.get("output_tokens") or meta.get("completion_tokens") or 0)
    if not found:
        return None, None, None
    return in_tok, out_tok, in_tok + out_tok
