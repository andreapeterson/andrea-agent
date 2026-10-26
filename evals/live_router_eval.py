"""Run explicitly selected Router cases against OpenAI only with live confirmation."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from openai import AsyncOpenAI

from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.router_agent import OpenAIRouter
from evals.router_eval import RouterEvalCase, format_router_report, load_router_cases, run_router_evaluation


def create_live_route_case(
    openai_client: AsyncOpenAI,
    model: str,
    renderer: PromptRenderer,
) -> Callable[[PromptContext, str], Awaitable[str]]:
    """Prepare one Router for reuse across the selected evaluation cases."""
    router = OpenAIRouter(openai_client, model=model)

    async def route_case(context: PromptContext, user_message: str) -> str:
        instructions = renderer.render_router(context)
        decision = await router.route(instructions=instructions, user_message=user_message)
        return decision.destination

    return route_case


async def _evaluate(cases: list[RouterEvalCase], api_key: str, model: str) -> dict[str, Any]:
    async with AsyncOpenAI(api_key=api_key) as openai_client:
        route_case = create_live_route_case(openai_client, model, PromptRenderer())
        return await run_router_evaluation(cases, route_case)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-live", action="store_true", required=True, help="Confirm network requests and API cost")
    parser.add_argument("--model", default=os.getenv("OPENAI_ROUTER_MODEL") or "gpt-6-luna")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id", action="append", help="Case to evaluate; repeat to select several")
    selection.add_argument("--all-cases", action="store_true", help="Explicitly select the entire Router dataset")
    args = parser.parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        parser.error("OPENAI_API_KEY is required for live evaluation")
    if not args.model.strip():
        parser.error("The Router model must not be blank")
    try:
        cases = load_router_cases()
    except (OSError, ValueError):
        parser.error("Unable to load the Router evaluation dataset")

    requested_ids = set(args.case_id or [])
    unknown_ids = requested_ids - {case.case_id for case in cases}
    if unknown_ids:
        parser.error("Unknown case IDs: " + ", ".join(sorted(unknown_ids)))
    selected_cases = [case for case in cases if args.all_cases or case.case_id in requested_ids]

    try:
        report = asyncio.run(_evaluate(selected_cases, api_key, args.model.strip()))
    except Exception:
        # Exception chains can contain credentials, prompts, or raw upstream responses.
        print("Live Router evaluation failed; no report was produced.", file=sys.stderr)
        return 1
    print(format_router_report(report))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
