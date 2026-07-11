"""DeepAgents Daily Assistant — agent factory and turn runner."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.config import GATED_TOOLS, AgentConfig, get_config
from agent.prompts import build_system_prompt
from agent.reviewer import build_reviewer_subagent

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AGENTS_MD = _PROJECT_ROOT / "AGENTS.md"


@dataclass
class ToolCallTrace:
    name: str
    args: dict[str, Any]
    result_preview: str | None = None


@dataclass
class AgentTurnResult:
    reply: str
    thread_id: str
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    interrupted: bool = False
    interrupt_payload: Any = None
    raw_messages: list[Any] = field(default_factory=list)


@dataclass
class DailyAssistant:
    """Wrapper around a compiled DeepAgent graph + MCP client lifecycle."""

    agent: Any
    mcp_client: MultiServerMCPClient
    config: AgentConfig
    checkpointer: MemorySaver
    tools: list[Any]

    async def aclose(self) -> None:
        """No persistent connection to close — MultiServerMCPClient opens a
        short-lived session per call (get_tools/tool invocation) rather than
        holding one open, so there is nothing to tear down here."""


def _mcp_server_config(cfg: AgentConfig) -> dict[str, Any]:
    """Build MultiServerMCPClient connection config.

    Supports SSE (default, matches docker-compose) and streamable HTTP.
    """
    transport = cfg.mcp_transport.lower()
    url = cfg.mcp_url
    if transport == "sse":
        return {"dev-tools": {"transport": "sse", "url": url}}
    if transport in ("http", "streamable_http", "streamable-http"):
        # Prefer /mcp for streamable HTTP if caller still points at /sse
        if url.rstrip("/").endswith("/sse"):
            url = url.rstrip("/")[: -len("/sse")] + "/mcp"
        return {"dev-tools": {"transport": "streamable_http", "url": url}}
    return {"dev-tools": {"transport": transport, "url": url}}


async def create_daily_assistant(
    config: AgentConfig | None = None,
) -> DailyAssistant:
    """Create the Daily Assistant DeepAgent wired to the developer-tools MCP server."""
    cfg = config or get_config()
    mcp_client = MultiServerMCPClient(_mcp_server_config(cfg))
    tools = await mcp_client.get_tools()

    interrupt_on = {name: True for name in GATED_TOOLS}
    checkpointer = MemorySaver()

    create_kwargs: dict[str, Any] = {
        "model": cfg.model,
        "tools": tools,
        "system_prompt": build_system_prompt(cfg.reference_date),
        "interrupt_on": interrupt_on,
        "subagents": [build_reviewer_subagent()],
        "checkpointer": checkpointer,
    }
    # Persistent project memory (DeepAgents loads AGENTS.md at startup)
    if _AGENTS_MD.exists():
        create_kwargs["memory"] = [str(_AGENTS_MD)]

    agent = create_deep_agent(**create_kwargs)

    return DailyAssistant(
        agent=agent,
        mcp_client=mcp_client,
        config=cfg,
        checkpointer=checkpointer,
        tools=tools,
    )


def _extract_tool_calls(messages: list[Any]) -> list[ToolCallTrace]:
    traces: list[ToolCallTrace] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name", "unknown")
                args = tc.get("args") or {}
            else:
                name = getattr(tc, "name", "unknown")
                args = getattr(tc, "args", {}) or {}
            traces.append(ToolCallTrace(name=name, args=dict(args)))

        # Attach result previews to matching tool messages
        if getattr(msg, "type", None) == "tool" or msg.__class__.__name__ == "ToolMessage":
            name = getattr(msg, "name", None) or "tool"
            content = getattr(msg, "content", "")
            preview = content if isinstance(content, str) else str(content)
            if len(preview) > 500:
                preview = preview[:500] + "…"
            # Update last matching trace without result
            for t in reversed(traces):
                if t.name == name and t.result_preview is None:
                    t.result_preview = preview
                    break
            else:
                traces.append(ToolCallTrace(name=name, args={}, result_preview=preview))
    return traces


def _last_ai_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        cls = msg.__class__.__name__
        if role in ("ai", "assistant") or cls in ("AIMessage", "AIMessageChunk"):
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif hasattr(block, "text"):
                        parts.append(block.text)
                return "".join(parts)
    return ""


def _get_interrupt(result: dict[str, Any] | Any) -> Any | None:
    if isinstance(result, dict):
        if result.get("__interrupt__"):
            return result["__interrupt__"]
        # LangGraph may nest state
        for key in ("__interrupt__", "interrupt"):
            if key in result:
                return result[key]
    interrupts = getattr(result, "interrupts", None)
    if interrupts:
        return interrupts
    return None


async def run_turn(
    assistant: DailyAssistant,
    user_message: str,
    *,
    thread_id: str | None = None,
    resume: dict[str, Any] | None = None,
) -> AgentTurnResult:
    """Run one user turn (or resume after HITL) and return reply + trace."""
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    if resume is not None:
        payload: Any = Command(resume=resume)
    else:
        payload = {"messages": [{"role": "user", "content": user_message}]}

    result = await assistant.agent.ainvoke(payload, config=config)

    interrupt = _get_interrupt(result)
    messages = []
    if isinstance(result, dict):
        messages = list(result.get("messages") or [])
    else:
        messages = list(getattr(result, "messages", []) or [])

    return AgentTurnResult(
        reply=_last_ai_text(messages) if not interrupt else (
            _last_ai_text(messages)
            or "A write action requires your approval before it can proceed."
        ),
        thread_id=tid,
        tool_calls=_extract_tool_calls(messages),
        interrupted=bool(interrupt),
        interrupt_payload=interrupt,
        raw_messages=messages,
    )


async def stream_turn(
    assistant: DailyAssistant,
    user_message: str,
    *,
    thread_id: str | None = None,
    resume: dict[str, Any] | None = None,
    on_event: Callable[[str, Any], Awaitable[None] | None] | None = None,
) -> AgentTurnResult:
    """Stream a turn, optionally invoking on_event(event_type, data)."""
    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    if resume is not None:
        payload: Any = Command(resume=resume)
    else:
        payload = {"messages": [{"role": "user", "content": user_message}]}

    collected_messages: list[Any] = []
    interrupt: Any = None

    async for event in assistant.agent.astream(
        payload,
        config=config,
        stream_mode=["messages", "updates"],
    ):
        # event shape varies by stream_mode; normalize loosely
        if isinstance(event, tuple) and len(event) == 2:
            mode, data = event
        else:
            mode, data = "updates", event

        if on_event is not None:
            maybe = on_event(str(mode), data)
            if hasattr(maybe, "__await__"):
                await maybe

        if mode == "messages":
            # (message_chunk, metadata)
            if isinstance(data, tuple) and data:
                collected_messages.append(data[0])
            else:
                collected_messages.append(data)
        elif mode == "updates" and isinstance(data, dict):
            if "__interrupt__" in data:
                interrupt = data["__interrupt__"]
            for node_data in data.values():
                if isinstance(node_data, dict) and "messages" in node_data:
                    collected_messages.extend(node_data["messages"])

    # Final state for authoritative messages / interrupt
    state = await assistant.agent.aget_state(config)
    values = getattr(state, "values", None) or {}
    if isinstance(values, dict) and values.get("messages"):
        collected_messages = list(values["messages"])
    tasks = getattr(state, "tasks", None) or ()
    for task in tasks:
        ints = getattr(task, "interrupts", None) or ()
        if ints:
            interrupt = ints
            break

    return AgentTurnResult(
        reply=_last_ai_text(collected_messages) if not interrupt else (
            _last_ai_text(collected_messages)
            or "A write action requires your approval before it can proceed."
        ),
        thread_id=tid,
        tool_calls=_extract_tool_calls(collected_messages),
        interrupted=bool(interrupt),
        interrupt_payload=interrupt,
        raw_messages=collected_messages,
    )


def approval_resume(decision: str = "approve") -> dict[str, Any]:
    """Build a Command resume payload for HITL decisions."""
    decision = decision.lower().strip()
    if decision in ("approve", "accept", "yes", "y"):
        return {"decisions": [{"type": "approve"}]}
    if decision in ("reject", "deny", "no", "n"):
        return {"decisions": [{"type": "reject"}]}
    return {"decisions": [{"type": decision}]}
