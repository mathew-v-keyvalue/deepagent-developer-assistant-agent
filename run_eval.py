#!/usr/bin/env python3
"""CLI entry point for bake-off evaluation runs."""

from __future__ import annotations

import argparse
import asyncio

from harness.runner import run_eval_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Daily Assistant golden tasks against the DeepAgents implementation"
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Only run tasks for this mode",
    )
    parser.add_argument("--task", type=str, default=None, help="Run a single task id")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeats per task (use ~5 for reliability scoring)",
    )
    parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="Do not auto-approve write actions (stop at HITL gate)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory for JSON result files",
    )
    args = parser.parse_args()

    asyncio.run(
        run_eval_suite(
            mode=args.mode,
            task_id=args.task,
            repeats=args.repeats,
            auto_approve=not args.no_auto_approve,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
