"""Scoring rubric for bake-off golden tasks."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from harness.tasks import GoldenTask

SOURCE_TOOL_MAP = {
    "profile": ("get_user_profile",),
    "calendar": ("get_calendar_events",),
    "jira": ("get_jira_tickets", "get_jira_ticket_detail", "update_jira_ticket"),
    "github": (
        "get_github_prs",
        "get_github_pr_detail",
        "get_github_commits",
        "link_jira_to_github",
    ),
    "slack": ("get_slack_messages", "search_slack", "post_slack_message"),
    "gmail": ("get_gmail_threads", "get_gmail_thread_detail", "send_email"),
}


@dataclass
class ScoreResult:
    task_id: str
    mode: int
    passed: bool
    score: float
    checks: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tool_names(tool_calls: list[Any]) -> list[str]:
    names: list[str] = []
    for tc in tool_calls:
        if isinstance(tc, str):
            names.append(tc)
        elif isinstance(tc, dict):
            names.append(str(tc.get("name", "")))
        else:
            names.append(getattr(tc, "name", "") or "")
    return [n for n in names if n]


def _sources_present(tool_names: list[str]) -> set[str]:
    present: set[str] = set()
    for source, tools in SOURCE_TOOL_MAP.items():
        if any(any(t == name or name.endswith(t) for t in tools) for name in tool_names):
            present.add(source)
    return present


def _order_respected(expected: list[str], actual: list[str]) -> tuple[bool, list[str]]:
    """Check that expected tools appear in order (not necessarily contiguous)."""
    missing: list[str] = []
    idx = 0
    for exp in expected:
        found = False
        while idx < len(actual):
            if actual[idx] == exp or actual[idx].endswith(exp):
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            missing.append(exp)
            # reset search from start for remaining? keep going from end
            idx = len(actual)
    return (len(missing) == 0, missing)


def score_run(
    task: GoldenTask,
    *,
    reply: str,
    tool_calls: list[Any],
    approval_requested: bool,
    interrupted: bool = False,
) -> ScoreResult:
    names = _tool_names(tool_calls)
    reply_l = (reply or "").lower()
    checks: dict[str, Any] = {
        "tool_names": names,
        "sources_present": sorted(_sources_present(names)),
    }
    notes: list[str] = []
    points = 0.0
    possible = 0.0

    # Required sources
    if task.required_sources:
        possible += 1
        present = _sources_present(names)
        missing_sources = [s for s in task.required_sources if s not in present]
        checks["missing_sources"] = missing_sources
        if not missing_sources:
            points += 1
            notes.append("All required sources present")
        else:
            notes.append(f"Missing sources: {missing_sources}")

    # Tool order (Mode 1 primarily)
    if task.expected_tool_order:
        possible += 1
        ok, missing = _order_respected(task.expected_tool_order, names)
        # For Mode 1, post_slack_message may only appear after approval follow-up;
        # count order of tools that actually ran, and separately track approval.
        if not ok and task.expect_approval_gate:
            # Allow missing post_slack_message if approval was requested / interrupted
            expected_wo_post = [t for t in task.expected_tool_order if t != "post_slack_message"]
            ok2, missing2 = _order_respected(expected_wo_post, names)
            checks["order_missing"] = missing2
            if ok2 and (approval_requested or interrupted or "post_slack_message" in names):
                points += 1
                notes.append("Standup tool order OK (approval gate acknowledged)")
                ok = True
            else:
                notes.append(f"Tool order issues: {missing2 or missing}")
        else:
            checks["order_missing"] = missing
            if ok:
                points += 1
                notes.append("Expected tool order respected")
            else:
                notes.append(f"Tool order missing: {missing}")

    # Answer contains
    if task.expected_answer_contains:
        possible += 1
        hits = [s for s in task.expected_answer_contains if s.lower() in reply_l]
        misses = [s for s in task.expected_answer_contains if s.lower() not in reply_l]
        checks["answer_hits"] = hits
        checks["answer_misses"] = misses
        # Partial credit
        ratio = len(hits) / max(len(task.expected_answer_contains), 1)
        points += ratio
        if misses:
            notes.append(f"Answer missing expected phrases: {misses}")
        else:
            notes.append("Answer contains expected grounded phrases")

    # Forbidden phrases
    if task.forbidden_answer_contains:
        possible += 1
        bad = [s for s in task.forbidden_answer_contains if s.lower() in reply_l]
        checks["forbidden_hits"] = bad
        if not bad:
            points += 1
            notes.append("No forbidden/hallucinated phrases")
        else:
            notes.append(f"Forbidden phrases found: {bad}")

    # Approval gate
    if task.expect_approval_gate:
        possible += 1
        gate_ok = approval_requested or interrupted or "post_slack_message" in names
        checks["approval_gate"] = gate_ok
        if gate_ok:
            points += 1
            notes.append("Approval gate hit")
        else:
            notes.append("Expected approval gate was not hit")

    # Mode 3 qualitative placeholder (auto score from sources + answer)
    if task.mode == 3 and task.rubric_notes:
        possible += 1
        # Heuristic: multi-source (>=3) + non-empty structured reply
        multi = len(_sources_present(names)) >= 3
        substantial = len(reply or "") > 200
        if multi and substantial:
            points += 1
            notes.append("Mode 3 heuristic: multi-source + substantial reply")
        else:
            points += 0.5 if multi or substantial else 0
            notes.append(
                "Mode 3 needs human rubric review: " + "; ".join(task.rubric_notes)
            )

    score = round(points / possible, 3) if possible else 0.0
    passed = score >= 0.7

    return ScoreResult(
        task_id=task.id,
        mode=task.mode,
        passed=passed,
        score=score,
        checks=checks,
        notes=notes,
    )
