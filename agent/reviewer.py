"""Reviewer sub-agent configuration for grounding validation."""

from __future__ import annotations

from typing import Any

from agent.prompts import REVIEWER_PROMPT


def build_reviewer_subagent() -> dict[str, Any]:
    """Return a declarative DeepAgents subagent spec for response review.

    The reviewer is intentionally tool-less (read-only). The main agent
    delegates via the built-in `task` tool when it wants a grounding check.
    """
    return {
        "name": "response-reviewer",
        "description": (
            "Reviews drafted assistant replies for data grounding, "
            "cross-source hallucination, temporal accuracy, and approval-gate "
            "compliance. Use before sending high-stakes multi-source answers."
        ),
        "system_prompt": REVIEWER_PROMPT,
        "tools": [],
    }
