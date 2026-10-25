"""Load human-labeled Router cases and score an injected routing implementation."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, get_args

from app.agents.prompt_context import PromptContext
from app.agents.router_agent import RouteDecision

DEFAULT_CASES_PATH = Path(__file__).with_name("router_cases.jsonl")
DESTINATIONS = get_args(RouteDecision.model_fields["destination"].annotation)


@dataclass
class RouterEvalCase:
    case_id: str
    user_message: str
    context: PromptContext
    expected_destination: str
    tags: list[str]
    reason: str


def _nonblank_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value.strip()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def load_router_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[RouterEvalCase]:
    """Skip whitespace-only lines; reject invalid rows with their file and line number. converts the JSONL file into validated Python objects"""
    cases = []
    seen_ids = set()
    fields = {"case_id", "user_message", "context", "expected_destination", "tags", "reason"}
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant) #json string into python dictionary
                if not isinstance(row, dict) or set(row) != fields:
                    raise ValueError(f"each case must contain exactly these fields: {', '.join(sorted(fields))}")
                case_id = _nonblank_string(row["case_id"], "case_id")
                if case_id in seen_ids:
                    raise ValueError(f"duplicate case_id: {case_id}")
                destination = row["expected_destination"]
                if destination not in DESTINATIONS:
                    raise ValueError(f"invalid expected_destination: {destination!r}")
                if not isinstance(row["tags"], list) or not row["tags"]:
                    raise ValueError("tags must be a nonempty list of nonblank strings")
                case = RouterEvalCase(
                    case_id=case_id,
                    user_message=_nonblank_string(row["user_message"], "user_message"),
                    context=PromptContext.model_validate(row["context"]),
                    expected_destination=destination,
                    tags=[_nonblank_string(tag, "tag") for tag in row["tags"]],
                    reason=_nonblank_string(row["reason"], "reason"),
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen_ids.add(case_id)
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
