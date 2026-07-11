#!/usr/bin/env python3
"""Interactive CLI for the DeepAgents Daily Assistant."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from agent.config import get_config
from agent.main import (
    DailyAssistant,
    approval_resume,
    create_daily_assistant,
    run_turn,
    stream_turn,
)

console = Console()


def _format_interrupt(payload: Any) -> str:
    try:
        return json.dumps(payload, default=str, indent=2)[:2000]
    except TypeError:
        return str(payload)[:2000]


async def _handle_hitl(assistant: DailyAssistant, thread_id: str, payload: Any) -> str:
    console.print(
        Panel(
            _format_interrupt(payload),
            title="[yellow]Human approval required[/yellow]",
            border_style="yellow",
        )
    )
    if assistant.config.auto_approve_writes:
        console.print("[dim]AUTO_APPROVE_WRITES=true — approving automatically.[/dim]")
        decision = "approve"
    else:
        choice = Prompt.ask(
            "Decision",
            choices=["approve", "reject"],
            default="approve",
        )
        decision = choice

    result = await run_turn(
        assistant,
        "",
        thread_id=thread_id,
        resume=approval_resume(decision),
    )
    # Nested interrupts (e.g. confirm_action after post)
    while result.interrupted:
        console.print(
            Panel(
                _format_interrupt(result.interrupt_payload),
                title="[yellow]Further approval required[/yellow]",
                border_style="yellow",
            )
        )
        if assistant.config.auto_approve_writes:
            nested = "approve"
        else:
            nested = Prompt.ask(
                "Decision",
                choices=["approve", "reject"],
                default="approve",
            )
        result = await run_turn(
            assistant,
            "",
            thread_id=thread_id,
            resume=approval_resume(nested),
        )
    return result.reply


async def chat_loop() -> None:
    cfg = get_config()
    console.print(
        Panel.fit(
            f"[bold]DeepAgents Daily Assistant[/bold]\n"
            f"Model: {cfg.model}\n"
            f"MCP: {cfg.mcp_url}\n"
            f"Reference date: {cfg.reference_date}\n"
            f"Type [cyan]exit[/cyan] to quit, [cyan]/standup[/cyan] for Mode 1 sample.",
            border_style="cyan",
        )
    )

    console.print("[dim]Connecting to MCP and creating agent…[/dim]")
    try:
        assistant = await create_daily_assistant(cfg)
    except Exception as exc:
        console.print(f"[red]Failed to create agent:[/red] {exc}")
        console.print(
            "Ensure the MCP server is running: [cyan]docker compose up --build[/cyan]"
        )
        sys.exit(1)

    console.print(
        f"[green]Ready.[/green] Loaded {len(assistant.tools)} MCP tools: "
        + ", ".join(sorted({t.name for t in assistant.tools})[:8])
        + ("…" if len(assistant.tools) > 8 else "")
    )

    thread_id = str(uuid.uuid4())

    try:
        while True:
            try:
                user_text = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/dim]")
                break

            user_text = (user_text or "").strip()
            if not user_text:
                continue
            if user_text.lower() in ("exit", "quit", "q"):
                break
            if user_text == "/standup":
                user_text = "Generate my standup update for today's 10 AM meeting"
            elif user_text == "/attention":
                user_text = "What needs my attention today?"
            elif user_text == "/prioritize":
                user_text = "I'm overcommitted this week, help me prioritize"

            console.print("[dim]Thinking…[/dim]")

            async def on_event(mode: str, data: Any) -> None:
                if mode != "updates" or not isinstance(data, dict):
                    return
                for node, node_data in data.items():
                    if node == "__interrupt__" or not isinstance(node_data, dict):
                        continue
                    for msg in node_data.get("messages") or []:
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        for tc in tool_calls:
                            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                            console.print(f"  [blue]→ tool[/blue] {name}")

            try:
                result = await stream_turn(
                    assistant,
                    user_text,
                    thread_id=thread_id,
                    on_event=on_event,
                )
            except Exception:
                # Fallback to non-streaming invoke
                result = await run_turn(assistant, user_text, thread_id=thread_id)

            reply = result.reply
            if result.interrupted:
                reply = await _handle_hitl(
                    assistant, thread_id, result.interrupt_payload
                )

            console.print()
            console.print(
                Panel(
                    Markdown(reply or "_(empty reply)_"),
                    title="[green]Assistant[/green]",
                    border_style="green",
                )
            )
            if result.tool_calls:
                names = [t.name for t in result.tool_calls]
                console.print(f"[dim]Tools used ({len(names)}): {', '.join(names)}[/dim]")
    finally:
        await assistant.aclose()


def main() -> None:
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
