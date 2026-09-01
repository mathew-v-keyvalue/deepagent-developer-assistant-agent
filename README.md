# DeepAgents Daily Assistant (Bake-off)

Daily assistant agent for software developers, built with the **DeepAgents SDK** (LangChain). Tools are provided by a shared, external **MCP server** (run separately, not part of this repo) exposing mock Jira, GitHub, Slack, Gmail, and Calendar APIs — every toolkit candidate in the bake-off connects to the same instance, unauthenticated, over SSE.

This is one of six toolkit implementations in the AI Agent Toolkit Bake-off. Everything except the SDK is held constant: one model, one tool set, one mock dataset, three task modes.

## Architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌───────────────────────────┐
│  CLI / Eval │────▶│  DeepAgents Agent    │────▶│  Shared MCP Server (SSE)  │
│  harness    │     │  + reviewer subagent │     │  developer-tools          │
│             │     │  + HITL interrupts   │     │  (external, no auth)      │
└─────────────┘     └──────────────────────┘     └───────────────────────────┘
```

**DeepAgents features used**

| Capability | How we use it |
|---|---|
| MCP tools | `langchain-mcp-adapters` → `MultiServerMCPClient` |
| Human-in-the-loop | `interrupt_on` for write + `confirm_action` tools |
| Subagents | `response-reviewer` for grounding checks |
| Memory / multi-turn | `MemorySaver` checkpointer + `thread_id` |
| Planning | Built-in `write_todos` for Mode 3 |
| Streaming | `astream` in the CLI |

## Quick start

### 1. Point at the shared MCP server

This repo does not run its own MCP server or database. Start (or confirm) the shared
`developer-tools` MCP server from its own project, then point this agent at it:

```bash
cp .env.example .env
# add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env
# confirm MCP_URL matches where the shared MCP server is running
```

- MCP SSE: `http://localhost:8081/sse` (default; no authentication required)

### 2. Install the agent

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Chat

```bash
python cli.py
```

Shortcuts inside the CLI:

- `/standup` — Mode 1 sample
- `/attention` — Mode 3 triage
- `/prioritize` — Mode 3 prioritization

### 4. Run evals

```bash
# All golden tasks, auto-approve writes
python run_eval.py

# Mode 2 only, 5 repeats for reliability
python run_eval.py --mode 2 --repeats 5

# Single task
python run_eval.py --task m2-jira-status
```

Results land in `eval_results/*.json` with tool traces, latency, token estimates, and rubric scores.

## Project layout

```
agent_sdk_deep_agent/
├── agent/               # create_deep_agent wiring, prompts, reviewer
├── harness/             # Shared runner, golden tasks, scoring, telemetry
├── cli.py               # Interactive chat
└── run_eval.py          # Eval entry point
```

The `developer-tools` MCP server (Jira/GitHub/Slack/Gmail/Calendar mock APIs + Postgres)
lives in its own project and is shared across all six toolkit implementations — it is not
part of this repo.

## MCP tools (fixed set)

| Tool | Type |
|---|---|
| `get_user_profile` | read |
| `get_jira_tickets` / `get_jira_ticket_detail` | read |
| `update_jira_ticket` | write — gated |
| `get_github_prs` / `get_github_pr_detail` / `get_github_commits` | read |
| `link_jira_to_github` | read |
| `get_slack_messages` / `search_slack` | read |
| `post_slack_message` | write — gated |
| `get_calendar_events` | read |
| `get_gmail_threads` / `get_gmail_thread_detail` | read |
| `send_email` | write — gated |
| `confirm_action` | write — gated (executes approved drafts) |
| `escalate_to_user` | action |

Write tools return a `pending_action_id`. The agent must wait for human approval (`interrupt_on`), then call `confirm_action`.

## Mock data (reference “today” = 2025-06-18, frozen — company Nimbus Labs)

- **Primary user:** Aisha Khan (`aisha.khan`), Backend Engineer, Asia/Kolkata
- **Jira:** 30 tickets across PROJ / INFRA / BUG
- **GitHub:** 30 PRs + 34 commits across 3 repos (`pulse-web`, `pulse-api`, `pulse-infra`; cross-linked via `PROJ-###:` / `INFRA-###:` / `BUG-###:` commit messages)
- **Slack:** 4 channels + DMs, 30 conversations (~67 messages)
- **Gmail:** 30 threads (PM, Tech Lead, QA, external client stakeholder)
- **Calendar:** 30 events — standups, Sprint 15 planning (today, 2 PM), 1:1s, code review sessions

Golden cross-link example: `PROJ-101` (SSO login) → PR `pulse-api#136` → Slack `#engineering` thread → Gmail `thread_1001` ("SSO login - client demo timing").

## Three task modes

| Mode | Example | Scoring |
|---|---|---|
| 1 Fixed steps | “Generate my standup update…” | Ordered tool checklist + approval gate |
| 2 Known scope | “What's the status of PROJ-101?” | Grounded vs golden phrases |
| 3 Open-ended | “I'm overcommitted this week…” | Multi-source + qualitative rubric |

## Common harness interface (§13)

```python
from agent.main import create_daily_assistant
from harness.runner import run_task
from harness.tasks import get_task

assistant = await create_daily_assistant()
tele = await run_task(assistant, get_task("m2-jira-status"), auto_approve=True)
print(tele.reply, tele.tool_calls, tele.pass_fail)
```

Every toolkit implementation should expose the same shape: conversation in → reply + ordered API trace out, with a HITL pause on writes.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `AGENT_MODEL` | `anthropic:claude-sonnet-4-6` | DeepAgents model string |
| `AGENT_TEMPERATURE` | `0` | Held constant for fairness |
| `MCP_URL` | `http://localhost:8081/sse` | Shared MCP server SSE endpoint (no auth) |
| `MCP_CLIENT_TRANSPORT` | `sse` | MCP client transport (`sse` or `streamable_http`) |
| `AUTO_APPROVE_WRITES` | `false` | CLI/eval HITL shortcut |
| `REFERENCE_DATE` | `2025-06-18` | Temporal anchor for prompts (must match the shared dataset's frozen `today`) |
| `PRIMARY_USER_ID` | `aisha.khan` | Default user context injected by the eval harness |

## Guardrails

1. No fabrication — claims must trace to tool results  
2. No public writes without HITL + `confirm_action`  
3. Escalate via `escalate_to_user` when data is missing  
4. Reviewer sub-agent available for grounding checks on Mode 3 answers  
# deepagent-developer-assistant-agent
