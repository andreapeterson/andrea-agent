"""Load human-labeled Router cases and score an injected routing implementation."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, get_args

from pydantic import BaseModel, ConfigDict, Field

from app.agents.prompt_context import PromptContext
from app.agents.router_agent import RouteDecision

DEFAULT_CASES_PATH = Path(__file__).with_name("router_cases.jsonl")
DESTINATIONS = get_args(RouteDecision.model_fields["destination"].annotation)


class RouterEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1)
    context: PromptContext
    expected_destination: str = Field(min_length=1)
    tags: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    reason: str = Field(min_length=1)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def load_router_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[RouterEvalCase]:
    """Skip whitespace-only lines; reject invalid rows with their file and line number. converts the JSONL file into validated Python objects"""
    cases = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant) #json string into python dictionary
                case = RouterEvalCase.model_validate(row)
                if case.case_id in seen_ids:
                    raise ValueError(f"duplicate case_id: {case.case_id}")
                if case.expected_destination not in DESTINATIONS:
                    raise ValueError(f"invalid expected_destination: {case.expected_destination!r}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen_ids.add(case.case_id)
            cases.append(case)
    return cases


async def run_router_evaluation(
    cases: Iterable[RouterEvalCase],
    route_case: Callable[[PromptContext, str], Awaitable[str]],
) -> dict[str, Any]:
    """Call once per case and compare destinations exactly; routing errors propagate."""
    results = []
    for case in cases:
        actual = await route_case(case.context, case.user_message)
        if not isinstance(actual, str):
            raise TypeError("route_case must return a destination string")
        results.append({
            "case_id": case.case_id,
            "expected_destination": case.expected_destination,
            "actual_destination": actual,
            "passed": actual == case.expected_destination,
            "tags": list(case.tags),
        })

    by_destination = {}
    for destination in DESTINATIONS:
        group = [result for result in results if result["expected_destination"] == destination]
        passed = sum(result["passed"] for result in group)
        by_destination[destination] = {
            "total": len(group),
            "passed": passed,
            "failed": len(group) - passed,
            "accuracy": passed / len(group) if group else 0.0,
        }
    total = len(results)
    passed = sum(result["passed"] for result in results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total if total else 0.0,
        "by_destination": by_destination,
        "results": results,
        "failures": [result for result in results if not result["passed"]],
    }


def format_router_report(report: dict[str, Any]) -> str:
    """Produce terminal-friendly counts and failures without storing any results."""
    lines = [
        "Router evaluation",
        f"Total: {report['total']}",
        f"Passed: {report['passed']}",
        f"Failed: {report['failed']}",
        f"Accuracy: {report['accuracy']:.1%}",
        "",
        "By destination:",
    ]
    for destination, group in report["by_destination"].items():
        lines.append(f"{destination}: {group['passed']}/{group['total']}")
    lines.extend(["", "Failures:"])
    for failure in report["failures"]:
        lines.append(
            f"- {failure['case_id']}: expected {failure['expected_destination']}, "
            f"received {failure['actual_destination']}"
        )
    if not report["failures"]:
        lines.append("- none")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--route-case", required=True, help="Injected async callable as module:function; no default model/client")
    args = parser.parse_args()
    module_name, separator, function_name = args.route_case.partition(":")
    if not separator or not module_name or not function_name:
        parser.error("--route-case must use module:function")
    route_case = getattr(import_module(module_name), function_name)
    if not callable(route_case):
        parser.error("--route-case must identify a callable")
    report = asyncio.run(run_router_evaluation(load_router_cases(args.cases), route_case))
    print(format_router_report(report))


if __name__ == "__main__":
    main()
