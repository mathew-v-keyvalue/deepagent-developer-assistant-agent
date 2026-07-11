"""Unified test runner — common interface for all bake-off toolkit implementations.

Entry point contract (§13):
  Input:  conversation (messages), task config (mode, user_id, datetime context)
  Output: agent reply + structured trace (API calls in order, tokens, latency)
  HITL:   pauses on write actions; auto-approve configurable for eval
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Literal

from agent.config import AgentConfig, get_config
from agent.main import (
    DailyAssistant,
    approval_resume,
    create_daily_assistant,
    run_turn,
)
from harness.scoring import score_run
from harness.tasks import GOLDEN_TASKS, GoldenTask, get_task, tasks_for_mode
from harness.telemetry import (
    RunTelemetry,
    ToolCallMetric,
    estimate_cost,
    extract_token_usage,
)


async def run_task(
    assistant: DailyAssistant,
    task: GoldenTask,
    *,
    auto_approve: bool | None = None,
    user_id: str | None = None,
    datetime_context: str | None = None,
) -> RunTelemetry:
    """Run a single golden task against the agent and return telemetry + score."""
    cfg = assistant.config
    approve = cfg.auto_approve_writes if auto_approve is None else auto_approve
    thread_id = str(uuid.uuid4())

    tele = RunTelemetry(
        task_id=task.id,
        mode=str(task.mode),
        thread_id=thread_id,
        model=cfg.model,
    )

    # Optional context preamble (kept out of golden prompt for fairness)
    context_bits = []
    if user_id or cfg.primary_user_id:
        context_bits.append(f"user_id={user_id or cfg.primary_user_id}")
    if datetime_context or cfg.reference_date:
        context_bits.append(f"reference_date={datetime_context or cfg.reference_date}")
    prompt = task.prompt
    if context_bits:
        prompt = f"[context: {', '.join(context_bits)}]\n{task.prompt}"

    try:
        result = await run_turn(assistant, prompt, thread_id=thread_id)

        # HITL loop
        approval_requested = result.interrupted
        safety = 0
        while result.interrupted and safety < 5:
            safety += 1
            tele.interrupted = True
            tele.approval_requested = True
            if not approve:
                # Stop at gate for manual scoring of Mode 1 step 9
                break
            result = await run_turn(
                assistant,
                "",
                thread_id=thread_id,
                resume=approval_resume("approve"),
            )

        # Optional natural-language approval follow-up (if agent asked in text)
        if (
            task.approval_followup
            and not tele.approval_requested
            and "post" in (result.reply or "").lower()
            and "?" in (result.reply or "")
        ):
            follow = await run_turn(
                assistant, task.approval_followup, thread_id=thread_id
            )
            approval_requested = approval_requested or follow.interrupted
            while follow.interrupted and approve and safety < 8:
                safety += 1
                tele.approval_requested = True
                follow = await run_turn(
                    assistant,
                    "",
                    thread_id=thread_id,
                    resume=approval_resume("approve"),
                )
            result = follow
            tele.approval_requested = tele.approval_requested or approval_requested

        tele.reply = result.reply or ""
        tele.interrupted = tele.interrupted or result.interrupted
        tele.approval_requested = tele.approval_requested or result.interrupted
        tele.tool_calls = [
            ToolCallMetric(
                name=tc.name,
                args=tc.args,
                result_preview=tc.result_preview,
            )
            for tc in result.tool_calls
        ]

        in_t, out_t, tot = extract_token_usage(result.raw_messages)
        tele.input_tokens = in_t
        tele.output_tokens = out_t
        tele.total_tokens = tot
        tele.estimated_cost_usd = estimate_cost(cfg.model, in_t, out_t)

        score = score_run(
            task,
            reply=tele.reply,
            tool_calls=tele.tool_calls,
            approval_requested=tele.approval_requested,
            interrupted=tele.interrupted,
        )
        tele.pass_fail = "pass" if score.passed else "fail"
        tele.score_details = score.to_dict()

    except Exception as exc:
        tele.error = f"{type(exc).__name__}: {exc}"
        tele.pass_fail = "error"
        tele.score_details = {"error": tele.error}

    tele.finish()
    return tele


async def run_eval_suite(
    *,
    mode: Literal[1, 2, 3] | None = None,
    task_id: str | None = None,
    repeats: int = 1,
    auto_approve: bool = True,
    config: AgentConfig | None = None,
    output_dir: str | Path = "eval_results",
) -> list[RunTelemetry]:
    """Run golden tasks (optionally filtered) and write JSON results."""
    cfg = config or get_config()
    # Force auto-approve for unattended eval unless caller overrides via env already
    if auto_approve:
        cfg = AgentConfig(
            model=cfg.model,
            temperature=cfg.temperature,
            mcp_url=cfg.mcp_url,
            mcp_transport=cfg.mcp_transport,
            reference_date=cfg.reference_date,
            primary_user_id=cfg.primary_user_id,
            auto_approve_writes=True,
        )

    if task_id:
        tasks = [get_task(task_id)]
    else:
        tasks = tasks_for_mode(mode)

    assistant = await create_daily_assistant(cfg)
    results: list[RunTelemetry] = []
    try:
        for task in tasks:
            for i in range(repeats):
                print(f"→ Running {task.id} (mode {task.mode}) repeat {i + 1}/{repeats}")
                tele = await run_task(assistant, task, auto_approve=auto_approve)
                results.append(tele)
                status = tele.pass_fail or "?"
                print(
                    f"  {status}  score={tele.score_details.get('score')}  "
                    f"latency_ms={tele.latency_ms}  tools={len(tele.tool_calls)}"
                )
    finally:
        await assistant.aclose()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    path = out / f"eval_{stamp}.json"
    payload = {
        "model": cfg.model,
        "reference_date": cfg.reference_date,
        "repeats": repeats,
        "mode_filter": mode,
        "results": [r.to_dict() for r in results],
        "summary": _summarize(results),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {path}")
    return results


def _summarize(results: list[RunTelemetry]) -> dict[str, Any]:
    by_mode: dict[str, list[RunTelemetry]] = {}
    for r in results:
        by_mode.setdefault(r.mode, []).append(r)

    summary: dict[str, Any] = {"total": len(results), "modes": {}}
    for mode, items in by_mode.items():
        passed = sum(1 for i in items if i.pass_fail == "pass")
        latencies = [i.latency_ms for i in items if i.latency_ms is not None]
        summary["modes"][mode] = {
            "count": len(items),
            "passed": passed,
            "pass_rate": round(passed / len(items), 3) if items else 0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "avg_score": round(
                sum(i.score_details.get("score", 0) or 0 for i in items) / len(items),
                3,
            )
            if items
            else 0,
        }
    return summary


# Sync helpers for scripts
def run_eval_suite_sync(**kwargs: Any) -> list[RunTelemetry]:
    return asyncio.run(run_eval_suite(**kwargs))
