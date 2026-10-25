"""Score specialist first actions offline using an injected async callable."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.agents.lookup_customer_tool import LookupCustomerArguments
from app.agents.prompt_context import PromptContext
from app.agents.record_intake_tool import RecordIntakeArguments
from app.agents.request_handoff_tool import RequestHandoffArguments
from app.agents.scheduling_tools import (
    BookSelectedAppointmentArguments,
    FindAppointmentSlotsArguments,
    SelectAppointmentSlotArguments,
)
from app.agents.search_policy_tool import SearchPolicyArguments

DEFAULT_CASES_PATH = Path(__file__).with_name("specialist_cases.jsonl")
TOOL_ARGUMENT_MODELS = {
    "lookup_customer": LookupCustomerArguments,
    "record_intake": RecordIntakeArguments,
    "search_policy": SearchPolicyArguments,
    "request_handoff": RequestHandoffArguments,
    "find_appointment_slots": FindAppointmentSlotsArguments,
    "select_appointment_slot": SelectAppointmentSlotArguments,
    "book_selected_appointment": BookSelectedAppointmentArguments,
}
SPECIALIST_TOOLS = {
    "verification": {"lookup_customer"},
    "intake": {"record_intake", "request_handoff"},
    "policy": {"search_policy"},
    "scheduling": {"find_appointment_slots", "select_appointment_slot", "book_selected_appointment"},
    "human_handoff": {"request_handoff"},
}


@dataclass(frozen=True)
class SpecialistEvalCase:
    case_id: str
    specialist: str
    user_message: str
    context: PromptContext
    expected_action: Literal["tool", "speech"]
    expected_tool: str | None
    expected_arguments: dict[str, object] | None
    tags: list[str]
    reason: str


@dataclass(frozen=True)
class SpecialistFirstAction: #what the specialist actually does when tested
    action: Literal["tool", "speech"]
    tool_name: str | None = None
    arguments: dict[str, object] | None = None

    def __post_init__(self) -> None:
        _validate_action(self.action, self.tool_name, self.arguments)


def _nonblank_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value.strip()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _json_arguments(arguments: dict[str, object] | None) -> str:
    # Object key order is irrelevant, but JSON types, nulls, and exact strings matter.
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _validate_action(action: str, tool: str | None, arguments: dict[str, object] | None) -> None:
    if action not in ("tool", "speech"):
        raise ValueError("action must be tool or speech")
    if action == "speech":
        if tool is not None or arguments is not None:
            raise ValueError("speech requires null tool and arguments")
    else:
        _nonblank_string(tool, "tool name")
        if not isinstance(arguments, dict):
            raise ValueError("tool action requires an arguments object")
        _json_arguments(arguments)


def load_specialist_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[SpecialistEvalCase]:
    """Validate labels with current argument models; never prepare or execute handlers."""
    fields = {
        "case_id", "specialist", "user_message", "context", "expected_action",
        "expected_tool", "expected_arguments", "tags", "reason",
    }
    cases = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(row, dict) or set(row) != fields:
                    raise ValueError(f"each case must contain exactly these fields: {', '.join(sorted(fields))}")
                case_id = _nonblank_string(row["case_id"], "case_id")
                if case_id in seen_ids:
                    raise ValueError(f"duplicate case_id: {case_id}")
                specialist = _nonblank_string(row["specialist"], "specialist")
                if specialist not in SPECIALIST_TOOLS:
                    raise ValueError(f"unsupported specialist: {specialist}")
                _validate_action(row["expected_action"], row["expected_tool"], row["expected_arguments"])
                if row["expected_action"] == "tool":
                    tool = row["expected_tool"]
                    if tool not in SPECIALIST_TOOLS[specialist]:
                        raise ValueError(f"unsupported tool for {specialist}: {tool}")
                    TOOL_ARGUMENT_MODELS[tool].model_validate(row["expected_arguments"])
                if not isinstance(row["tags"], list) or not row["tags"]:
                    raise ValueError("tags must be a nonempty list of nonblank strings")
                case = SpecialistEvalCase(
                    case_id=case_id,
                    specialist=specialist,
                    user_message=_nonblank_string(row["user_message"], "user_message"),
                    context=PromptContext.model_validate(row["context"]),
                    expected_action=row["expected_action"],
                    expected_tool=row["expected_tool"],
                    expected_arguments=row["expected_arguments"],
                    tags=[_nonblank_string(tag, "tag") for tag in row["tags"]],
                    reason=_nonblank_string(row["reason"], "reason"),
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen_ids.add(case_id)
            cases.append(case)
    return cases


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for name in ("action", "tool_name", "arguments", "complete_case"):
        applicable = [result["checks"][name] for result in results if result["checks"][name] is not None]
        passed = sum(applicable)
        metrics[name] = {
            "passed": passed,
            "total": len(applicable),
            "accuracy": passed / len(applicable) if applicable else 0.0,
        }
    passed = metrics["complete_case"]["passed"]
    return {"total": len(results), "passed": passed, "failed": len(results) - passed, "metrics": metrics}


async def run_specialist_evaluation(
    cases: Iterable[SpecialistEvalCase],
    run_case: Callable[[SpecialistEvalCase], Awaitable[SpecialistFirstAction]],
) -> dict[str, Any]:
    """Observe one action per case; score labels without calling agents or tools."""
    results = []
    for case in cases:
        observed = await run_case(case)
        if not isinstance(observed, SpecialistFirstAction):
            raise TypeError("run_case must return SpecialistFirstAction")
        expects_tool = case.expected_action == "tool"
        checks = {
            "action": observed.action == case.expected_action,
            "tool_name": observed.tool_name == case.expected_tool if expects_tool else None,
            "arguments": _json_arguments(observed.arguments) == _json_arguments(case.expected_arguments) if expects_tool else None,
        }
        checks["complete_case"] = all(check for check in checks.values() if check is not None)
        failures = []
        if not checks["action"]:
            failures.append(f"action: expected {case.expected_action}, received {observed.action}")
        if checks["tool_name"] is False:
            failures.append(f"tool_name: expected {case.expected_tool}, received {observed.tool_name}")
        if checks["arguments"] is False:
            failures.append(f"arguments: expected {_json_arguments(case.expected_arguments)}, received {_json_arguments(observed.arguments)}")
        results.append({
            "case_id": case.case_id,
            "specialist": case.specialist,
            "expected_action": case.expected_action,
            "actual_action": observed.action,
            "expected_tool": case.expected_tool,
            "actual_tool": observed.tool_name,
            "expected_arguments": case.expected_arguments,
            "actual_arguments": observed.arguments,
            "checks": checks,
            "passed": checks["complete_case"],
            "tags": list(case.tags),
            "failure_details": failures,
        })
    return {
        **_summarize(results),
        "by_specialist": {
            specialist: _summarize([result for result in results if result["specialist"] == specialist])
            for specialist in SPECIALIST_TOOLS
        },
        "results": results,
        "failures": [result for result in results if not result["passed"]],
    }


def format_specialist_report(report: dict[str, Any]) -> str:
    lines = [
        "Specialist evaluation", f"Total: {report['total']}",
        f"Passed: {report['passed']}", f"Failed: {report['failed']}", "", "Scores:",
    ]
    for name, metric in report["metrics"].items():
        lines.append(f"{name}: {metric['passed']}/{metric['total']} ({metric['accuracy']:.1%})")
    lines.extend(["", "By specialist:"])
    for specialist, summary in report["by_specialist"].items():
        lines.append(f"{specialist}: {summary['passed']}/{summary['total']}")
    lines.extend(["", "Failures:"])
    for failure in report["failures"]:
        lines.append(f"- {failure['case_id']}: {'; '.join(failure['failure_details'])}")
    if not report["failures"]:
        lines.append("- none")
    return "\n".join(lines)
