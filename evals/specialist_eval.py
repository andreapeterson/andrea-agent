"""Score specialist first actions offline using an injected async callable."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class SpecialistEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    specialist: str = Field(min_length=1)
    user_message: str = Field(min_length=1)
    context: PromptContext
    expected_action: Literal["tool", "speech"]
    expected_tool: str | None = Field(min_length=1)
    expected_arguments: dict[str, object] | None
    tags: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after") #expected_action, expected_tool, and expected_arguments must be consistent with each other. after individual field validation and model is created, check combo
    def validate_expected_action(self) -> Self:
        if self.expected_action == "speech":
            if self.expected_tool is not None or self.expected_arguments is not None:
                raise ValueError("speech requires null tool and arguments")
        else:
            if self.expected_tool is None:
                raise ValueError("tool action requires a nonblank expected_tool")
            if self.expected_arguments is None:
                raise ValueError("tool action requires an expected_arguments dictionary")
        return self


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
    cases = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
                case = SpecialistEvalCase.model_validate(row)
                if case.case_id in seen_ids:
                    raise ValueError(f"duplicate case_id: {case.case_id}")
                if case.specialist not in SPECIALIST_TOOLS:
                    raise ValueError(f"unsupported specialist: {case.specialist}")
                if case.expected_action == "tool":
                    if case.expected_tool not in SPECIALIST_TOOLS[case.specialist]:
                        raise ValueError(f"unsupported tool for {case.specialist}: {case.expected_tool}")
                    TOOL_ARGUMENT_MODELS[case.expected_tool].model_validate(case.expected_arguments)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen_ids.add(case.case_id)
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
        if metric["total"] == 0:
            lines.append(f"{name}: N/A (0 applicable cases)")
        else:
            lines.append(f"{name}: {metric['passed']}/{metric['total']} ({metric['accuracy']:.1%})")
    lines.extend(["", "By specialist:"])
    for specialist, summary in report["by_specialist"].items():
        lines.append(f"{specialist}: {summary['passed']}/{summary['total']}")
    lines.extend(["", "Failures:"])
    for failure in report["failures"]:
        details = list(failure["failure_details"])
        if failure["expected_action"] == "speech" and failure["actual_action"] == "tool":
            details.append(f"tool_name: received {failure['actual_tool']}")
            details.append(f"arguments: received {_json_arguments(failure['actual_arguments'])}")
        lines.append(f"- {failure['case_id']}: {'; '.join(details)}")
    if not report["failures"]:
        lines.append("- none")
    return "\n".join(lines)
