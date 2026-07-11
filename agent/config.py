"""Configuration for the Daily Assistant DeepAgent."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AgentConfig:
    model: str = os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-4-6")
    temperature: float = float(os.environ.get("AGENT_TEMPERATURE", "0"))
    mcp_url: str = os.environ.get("MCP_URL", "http://localhost:8081/sse")
    mcp_transport: str = os.environ.get("MCP_CLIENT_TRANSPORT", "sse")
    reference_date: str = os.environ.get("REFERENCE_DATE", "2025-06-18")
    primary_user_id: str = os.environ.get("PRIMARY_USER_ID", "aisha.khan")
    auto_approve_writes: bool = (
        os.environ.get("AUTO_APPROVE_WRITES", "false").lower() in ("1", "true", "yes")
    )


WRITE_TOOLS = (
    "post_slack_message",
    "send_email",
    "update_jira_ticket",
)

# confirm_action executes a previously approved draft — also gate it
GATED_TOOLS = WRITE_TOOLS + ("confirm_action",)


def get_config() -> AgentConfig:
    return AgentConfig()
