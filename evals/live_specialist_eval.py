"""Observe specialist first actions against OpenAI only with live confirmation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from openai import AsyncOpenAI

from app.agents.lookup_customer_tool import LOOKUP_CUSTOMER_TOOL
from app.agents.prompt_renderer import PromptRenderer
from app.agents.record_intake_tool import RECORD_INTAKE_TOOL
from app.agents.request_handoff_tool import REQUEST_HANDOFF_TOOL
from app.agents.scheduling_tools import BOOK_SELECTED_APPOINTMENT_TOOL, FIND_APPOINTMENT_SLOTS_TOOL, SELECT_APPOINTMENT_SLOT_TOOL
from app.agents.search_policy_tool import SEARCH_POLICY_TOOL
from evals.specialist_eval import (
    SpecialistEvalCase,
    SpecialistFirstAction,
    format_specialist_report,
    load_specialist_cases,
    run_specialist_evaluation,
)


def create_live_run_case(
    openai_client: AsyncOpenAI,
    model: str,
    renderer: PromptRenderer,
) -> Callable[[SpecialistEvalCase], Awaitable[SpecialistFirstAction]]:
    """Use production prompts and tool definitions without preparing any handlers."""
    specialists = {
        "verification": (renderer.render_verification, [LOOKUP_CUSTOMER_TOOL]),
        "intake": (renderer.render_intake, [RECORD_INTAKE_TOOL, REQUEST_HANDOFF_TOOL]),
        "policy": (renderer.render_policy, [SEARCH_POLICY_TOOL]),
        "scheduling": (renderer.render_scheduling, [FIND_APPOINTMENT_SLOTS_TOOL, SELECT_APPOINTMENT_SLOT_TOOL, BOOK_SELECTED_APPOINTMENT_TOOL]),
        "human_handoff": (renderer.render_human_handoff, [REQUEST_HANDOFF_TOOL]),
    }

    async def run_case(case: SpecialistEvalCase) -> SpecialistFirstAction:
        render, tools = specialists[case.specialist]
        instructions = render(case.context)
        response = await openai_client.responses.create(
            model=model,
            instructions=instructions,
            input=[{"role": "user", "content": case.user_message}],
            tools=tools,
            parallel_tool_calls=False,
        )
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            raise ValueError("Specialist response is missing output")
        tool_calls = [item for item in output if getattr(item, "type", None) == "function_call"]
        if len(tool_calls) > 1:
            raise ValueError("Specialist response contains multiple function calls")
        if tool_calls:
            tool_name = getattr(tool_calls[0], "name", None)
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("Specialist function call requires a nonblank tool name")
            raw_arguments = getattr(tool_calls[0], "arguments", None)
            if not isinstance(raw_arguments, str):
                raise ValueError("Specialist function arguments must be JSON text")
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Specialist function arguments must be a JSON object")
            # Capture the observation for scoring; never execute or repair the request.
            return SpecialistFirstAction(action="tool", tool_name=tool_name, arguments=arguments)

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Specialist response contains neither a function call nor speech")
        return SpecialistFirstAction(action="speech")

    return run_case


async def _evaluate(cases: list[SpecialistEvalCase], api_key: str, model: str) -> dict[str, Any]:
    async with AsyncOpenAI(api_key=api_key) as openai_client:
        run_case = create_live_run_case(openai_client, model, PromptRenderer())
        return await run_specialist_evaluation(cases, run_case)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-live", action="store_true", required=True, help="Confirm network requests and API cost")
    parser.add_argument("--model", default=os.getenv("OPENAI_SPECIALIST_MODEL") or "gpt-6-luna")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id", action="append", help="Case to evaluate; repeat to select several")
    selection.add_argument("--all-cases", action="store_true", help="Explicitly select the entire specialist dataset")
    args = parser.parse_args(argv)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        parser.error("OPENAI_API_KEY is required for live evaluation")
    if not args.model.strip():
        parser.error("The specialist model must not be blank")
    try:
        cases = load_specialist_cases()
    except (OSError, ValueError):
        parser.error("Unable to load the specialist evaluation dataset")

    requested_ids = set(args.case_id or [])
    unknown_ids = requested_ids - {case.case_id for case in cases}
    if unknown_ids:
        parser.error("Unknown case IDs: " + ", ".join(sorted(unknown_ids)))
    selected_cases = [case for case in cases if args.all_cases or case.case_id in requested_ids]

    try:
        report = asyncio.run(_evaluate(selected_cases, api_key, args.model.strip()))
    except Exception:
        # Exception chains can contain credentials, prompts, or raw upstream responses.
        print("Live specialist evaluation failed; no report was produced.", file=sys.stderr)
        return 1
    print(format_specialist_report(report))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
