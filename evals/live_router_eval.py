"""Run explicitly selected Router cases against OpenAI only with live confirmation."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from app.agents.decisions_router import DECISIONS_ROUTER_MODEL
from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.router_factory import RouterBackend, create_router
from evals.failure_attribution import (
    attribute_report_failures,
    format_failure_attribution_report,
)
from evals.latency import (
    LatencySummary,
    calculate_latency_speedup,
    format_latency_comparison,
    format_router_latency_report,
    summarize_latencies,
)
from evals.router_eval import (
    RouterEvalCase,
    format_router_report,
    load_router_cases,
    run_router_evaluation,
)


def create_live_route_case(
    openai_client: AsyncOpenAI,
    model: str,
    renderer: PromptRenderer,
    *,
    router_backend: RouterBackend = "responses",
    duration_samples: list[float] | None = None,
    clock: Callable[[], float] | None = None,
) -> Callable[[PromptContext, str], Awaitable[str]]:
    """Prepare one Router for reuse across the selected evaluation cases."""
    router = create_router(openai_client, backend=router_backend, model=model)
    timer = perf_counter if clock is None else clock

    async def route_case(context: PromptContext, user_message: str) -> str:
        instructions = renderer.render_router(context)
        started = timer()
        decision = await router.route(instructions=instructions, user_message=user_message)
        elapsed_ms = (timer() - started) * 1000
        if duration_samples is not None:
            duration_samples.append(elapsed_ms)
        return decision.destination

    return route_case


async def _evaluate(
    cases: list[RouterEvalCase], api_key: str, model: str, router_backend: RouterBackend = "responses",
    duration_samples: list[float] | None = None,
) -> dict[str, Any]:
    async with AsyncOpenAI(api_key=api_key) as openai_client:
        route_case = create_live_route_case(
            openai_client, model, PromptRenderer(), router_backend=router_backend,
            duration_samples=duration_samples,
        )
        return await run_router_evaluation(cases, route_case)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-live", action="store_true", required=True, help="Confirm network requests and API cost")
    parser.add_argument("--router-backend", choices=["responses", "decisions", "both"], default="responses")
    parser.add_argument("--model", help="Responses model override; Decisions supports only gpt-6-luna")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id", action="append", help="Case to evaluate; repeat to select several")
    selection.add_argument("--tag", help="Select every case containing this exact tag")
    selection.add_argument("--all-cases", action="store_true", help="Explicitly select the entire Router dataset")
    args = parser.parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        parser.error("OPENAI_API_KEY is required for live evaluation")
    model = args.model if args.model is not None else (
        DECISIONS_ROUTER_MODEL if args.router_backend == "decisions"
        else os.getenv("OPENAI_ROUTER_MODEL") or "gpt-6-luna"
    )
    if not model.strip():
        parser.error("The Router model must not be blank")
    model = model.strip()
    if args.router_backend == "decisions" and model != DECISIONS_ROUTER_MODEL:
        parser.error("The Decisions Router supports only gpt-6-luna")
    try:
        cases = load_router_cases()
    except (OSError, ValueError):
        parser.error("Unable to load the Router evaluation dataset")

    requested_ids = set(args.case_id or [])
    unknown_ids = requested_ids - {case.case_id for case in cases}
    if unknown_ids:
        parser.error("Unknown case IDs: " + ", ".join(sorted(unknown_ids)))
    if args.tag is not None:
        known_tags = {"security", "prompt_injection", "tool_output_injection"} | {tag for case in cases for tag in case.tags}
        if args.tag not in known_tags:
            parser.error("Unknown evaluation tag")
        selected_cases = [case for case in cases if args.tag in case.tags]
        if not selected_cases:
            parser.error("No evaluation cases match the selected tag")
    else:
        selected_cases = [case for case in cases if args.all_cases or case.case_id in requested_ids]

    if not selected_cases:
        parser.error("No Router evaluation cases were selected")

    backends = ("responses", "decisions") if args.router_backend == "both" else (args.router_backend,)
    rendered_runs: list[str] = []
    summaries: dict[str, LatencySummary] = {}
    any_failed = False
    timing_incomplete = False
    # The same selection is reused, in dataset order. Finish one backend before
    # starting the next; no concurrent model requests or background benchmarks.
    for backend in backends:
        backend_model = DECISIONS_ROUTER_MODEL if backend == "decisions" else model
        duration_samples: list[float] = []
        try:
            report = asyncio.run(_evaluate(selected_cases, api_key, backend_model, backend, duration_samples))
        except (OpenAIError, OSError, RuntimeError, TypeError, ValueError, AttributeError, KeyError):
            # Never stringify upstream exceptions, which may contain content.
            print("Live Router evaluation failed; no report was produced.", file=sys.stderr)
            return 1
        try:
            attributions = attribute_report_failures(report, eval_type="router")
            localization = format_failure_attribution_report(attributions) if attributions else None
            text = format_router_report(report)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            print("Live Router failure localization failed; no report was produced.", file=sys.stderr)
            return 1
        output = f"Router backend: {backend}; model: {backend_model}\n{text}"
        if args.router_backend == "both" and backend == "decisions" and model != backend_model:
            output += "\n\nDecisions uses its fixed supported model; models differ."
        if localization is not None:
            output += f"\n\n{localization}"
        if report["failed"]:
            any_failed = True
        else:
            try:
                if report["total"] != len(selected_cases) or len(duration_samples) != len(selected_cases):
                    raise ValueError("A completed benchmark requires one sample per selected case.")
                summary = summarize_latencies(duration_samples)
                output += "\n\n" + format_router_latency_report(
                    summary, backend=backend, model=backend_model,
                    passed=report["passed"], failed=report["failed"],
                )
                summaries[backend] = summary
            except (TypeError, ValueError, OverflowError):
                timing_incomplete = True
        rendered_runs.append(output)

    comparison = None
    if args.router_backend == "both" and not any_failed and not timing_incomplete:
        try:
            comparison = format_latency_comparison(
                calculate_latency_speedup(summaries["responses"], summaries["decisions"]),
                responses_model=model, decisions_model=DECISIONS_ROUTER_MODEL,
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            timing_incomplete = True
    print("\n\n".join(rendered_runs))
    if timing_incomplete:
        message = (
            "Router latency comparison incomplete; no comparison was produced."
            if args.router_backend == "both"
            else "Router latency benchmark incomplete; no latency summary was produced."
        )
        print(message, file=sys.stderr)
        return 1
    if comparison is not None:
        print()
        print(comparison)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
