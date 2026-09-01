"""Golden task definitions for the three bake-off modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal[1, 2, 3]


@dataclass(frozen=True)
class GoldenTask:
    id: str
    mode: Mode
    prompt: str
    description: str
    # Mode 1: ordered tool-name checklist (substrings / exact names)
    expected_tool_order: list[str] = field(default_factory=list)
    # Sources that must appear (jira, github, slack, gmail, calendar, profile)
    required_sources: list[str] = field(default_factory=list)
    # Substrings that should appear in a grounded answer
    expected_answer_contains: list[str] = field(default_factory=list)
    # Substrings that must NOT appear (hallucination traps)
    forbidden_answer_contains: list[str] = field(default_factory=list)
    # Whether a write-tool interrupt / approval is expected
    expect_approval_gate: bool = False
    # Follow-up user message after approval prompt (Mode 1)
    approval_followup: str | None = None
    # Qualitative notes for Mode 3 human scoring
    rubric_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


GOLDEN_TASKS: list[GoldenTask] = [
    # ------------------------------------------------------------------
    # Mode 1 — Fixed steps (standup)
    # ------------------------------------------------------------------
    GoldenTask(
        id="m1-standup-basic",
        mode=1,
        prompt="Generate my standup update for today's 10 AM meeting",
        description="Full 10-step standup procedure with Slack approval gate",
        expected_tool_order=[
            "get_user_profile",
            "get_calendar_events",
            "get_jira_tickets",
            "link_jira_to_github",
            "get_github_pr_detail",
            "get_slack_messages",
            "get_gmail_threads",
            "post_slack_message",
        ],
        required_sources=["profile", "calendar", "jira", "github", "slack", "gmail"],
        expected_answer_contains=["Yesterday", "Today", "Blockers"],
        expect_approval_gate=True,
        approval_followup="Yes, post it to #standup",
        rubric_notes=[
            "All 5 data sources checked in order",
            "Structured Yesterday/Today/Blockers",
            "HITL before Slack post",
        ],
    ),
    # ------------------------------------------------------------------
    # Mode 2 — Focused queries
    # ------------------------------------------------------------------
    GoldenTask(
        id="m2-jira-status",
        mode=2,
        prompt="What's the status of PROJ-101?",
        description="Single-ticket grounded status lookup",
        expected_tool_order=["get_jira_ticket_detail"],
        required_sources=["jira"],
        expected_answer_contains=["PROJ-101", "In Review", "SSO"],
        forbidden_answer_contains=["PROJ-9999"],
    ),
    GoldenTask(
        id="m2-afternoon-meetings",
        mode=2,
        prompt="Do I have any meetings this afternoon?",
        description="Calendar filter for after 12 PM today",
        expected_tool_order=["get_calendar_events"],
        required_sources=["calendar"],
        expected_answer_contains=["Sprint", "2"],
        # Incident Follow-up (12:30) and Code Review Session (16:00) also afternoon — either is fine
    ),
    GoldenTask(
        id="m2-prs-for-review",
        mode=2,
        prompt="Show me PRs waiting for my review",
        description="Open PRs where Aisha is a reviewer",
        expected_tool_order=["get_github_prs"],
        required_sources=["github"],
        expected_answer_contains=["#"],
    ),
    GoldenTask(
        id="m2-slack-pm-mentions",
        mode=2,
        prompt="What did the PM mention me about in Slack yesterday?",
        description="Sarah Lin (PM) mentions of Aisha since yesterday",
        expected_tool_order=["get_slack_messages"],
        required_sources=["slack"],
        expected_answer_contains=["Sarah"],
    ),
    GoldenTask(
        id="m2-unread-product-email",
        mode=2,
        prompt="Any unread emails from the product team?",
        description="Unread work emails (PM / product)",
        expected_tool_order=["get_gmail_threads"],
        required_sources=["gmail"],
        expected_answer_contains=["unread"],
    ),
    # ------------------------------------------------------------------
    # Mode 3 — Open-ended
    # ------------------------------------------------------------------
    GoldenTask(
        id="m3-overcommitted",
        mode=3,
        prompt="I'm overcommitted this week, help me prioritize",
        description="Multi-source prioritization with conflicts",
        required_sources=["calendar", "jira", "github", "slack"],
        expected_answer_contains=["PROJ-101"],
        rubric_notes=[
            "Checks calendar, Jira, GitHub, Slack",
            "Identifies conflicts / overdue / blocked PRs",
            "Proposes ranked priorities with reasoning",
        ],
    ),
    GoldenTask(
        id="m3-sprint-prep",
        mode=3,
        prompt="Prep me for the 2 PM sprint planning meeting",
        description="Meeting briefing across sprint sources",
        required_sources=["calendar", "jira", "github", "slack"],
        expected_answer_contains=["Sprint Planning"],
        rubric_notes=[
            "Finds 2 PM sprint planning on calendar",
            "Summarizes sprint tickets / PR status / blockers",
            "Actionable briefing with questions to raise",
        ],
    ),
    GoldenTask(
        id="m3-attention-today",
        mode=3,
        prompt="What needs my attention today?",
        description="Cross-source triage for today",
        required_sources=["calendar", "jira", "github", "slack", "gmail"],
        expected_answer_contains=["#"],
        rubric_notes=[
            "Urgent / Important / Nice-to-have ranking",
            "Mentions blocked PRs, overdue tickets, meeting prep",
            "Grounded in tool results only",
        ],
    ),
]


def tasks_for_mode(mode: Mode | None = None) -> list[GoldenTask]:
    if mode is None:
        return list(GOLDEN_TASKS)
    return [t for t in GOLDEN_TASKS if t.mode == mode]


def get_task(task_id: str) -> GoldenTask:
    for t in GOLDEN_TASKS:
        if t.id == task_id:
            return t
    raise KeyError(f"Unknown task id: {task_id}")
