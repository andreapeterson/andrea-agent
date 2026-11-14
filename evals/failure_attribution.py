"""Deterministic failure localization from safe evaluation result fields only."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

EvalType = Literal["router", "specialist", "response", "conversation"]
Component = Literal[
    "router_selection", "specialist_action", "tool_selection", "tool_arguments",
    "response_generation", "orchestration", "conversation_state",
]
COMPONENTS = get_args(Component)
REVIEW_TARGETS = {
    "router_selection": "Review router prompt, routing labels, and routing fixtures.",
    "specialist_action": "Review specialist prompt and speech-versus-tool rules.",
    "tool_selection": "Review specialist tool descriptions and tool eligibility rules.",
    "tool_arguments": "Review tool schema, argument descriptions, and extraction examples.",
    "response_generation": "Review specialist response prompt and failed response criteria.",
    "orchestration": "Review routing continuity and specialist/tool sequencing.",
    "conversation_state": "Review handler state updates and terminal phase transitions.",
}
SPECIALIST_CHECKS = (
    ("action", "specialist_action"), ("tool_name", "tool_selection"), ("arguments", "tool_arguments"),
)
CONVERSATION_CHECKS = (
    ("specialist_sequence", "orchestration"), ("tool_sequence", "orchestration"),
    ("final_phase", "conversation_state"), ("verified_customer", "conversation_state"),
    ("booking_confirmed", "conversation_state"), ("handoff_queued", "conversation_state"),
)


class FailureAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True, strict=True)

    case_id: str = Field(min_length=1)
    eval_type: EvalType
    component: Component
    observed_failure: str = Field(min_length=1)
    suggested_review_target: str = Field(min_length=1)

# small helpers validate untrusted dictionaries (even though we produced them)
def _mapping(value: object, label: str) -> Mapping[str, Any]: #confirms dictionary-like 
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _text(row: Mapping[str, Any], field: str) -> str: #confirms string and strips whitespace
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value.strip()


def _flag(row: Mapping[str, Any], field: str, *, nullable: bool = False) -> bool | None: #confirms boolean or None if nullable
    if field not in row:
        raise ValueError(f"Missing required field: {field}")
    value = row[field]
    if type(value) is not bool and not (nullable and value is None):
        raise ValueError(f"{field} must be a boolean" + (" or None" if nullable else ""))
    return value


def _result(result: object) -> tuple[Mapping[str, Any], str, bool]: #extracts common fields from every eval 
    row = _mapping(result, "Result")
    return row, _text(row, "case_id"), _flag(row, "passed")


def _attribution(case_id: str, eval_type: EvalType, component: Component, observation: str) -> FailureAttribution:
    return FailureAttribution(
        case_id=case_id, eval_type=eval_type, component=component,
        observed_failure=observation, suggested_review_target=REVIEW_TARGETS[component],
    )


def attribute_router_result(result: Mapping[str, Any]) -> tuple[FailureAttribution, ...]:
    row, case_id, passed = _result(result)
    expected = _text(row, "expected_destination")
    actual = row.get("actual_destination")
    # The Router scorer permits any returned string, including an empty one.
    if not isinstance(actual, str):
        raise TypeError("actual_destination must be a string")
    if passed != (expected == actual): #reports dont contradict itself
        raise ValueError("Router passed flag disagrees with the destination comparison")
    return () if passed else (_attribution(
        case_id, "router", "router_selection", "Failure observed in Router destination selection.",
    ),)


def _attribute_checks(
    result: Mapping[str, Any], eval_type: EvalType, definitions: tuple[tuple[str, Component], ...],
) -> tuple[FailureAttribution, ...]:
    row, case_id, passed = _result(result)
    checks = _mapping(row.get("checks"), "checks")
    values = {
        name: _flag(checks, name, nullable=eval_type == "specialist" and name in {"tool_name", "arguments"})
        for name, _ in definitions
    }
    if eval_type == "specialist" and (values["tool_name"] is None) != (values["arguments"] is None):
        raise ValueError("Specialist tool_name and arguments checks must both be applicable or both be None")
    complete = _flag(checks, "complete_case")
    if passed != complete or complete != all(value for value in values.values() if value is not None):
        raise ValueError("passed and complete_case must agree with the individual checks")
    return tuple(
        _attribution(case_id, eval_type, component, f"Failure observed in {eval_type} check: {name}.")
        for name, component in definitions if values[name] is False
    )


def attribute_specialist_result(result: Mapping[str, Any]) -> tuple[FailureAttribution, ...]:
    return _attribute_checks(result, "specialist", SPECIALIST_CHECKS)


def attribute_conversation_result(result: Mapping[str, Any]) -> tuple[FailureAttribution, ...]:
    return _attribute_checks(result, "conversation", CONVERSATION_CHECKS)


def attribute_response_result(result: Mapping[str, Any]) -> tuple[FailureAttribution, ...]:
    row, case_id, passed = _result(result)
    criteria = row.get("criteria_results")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("criteria_results must be a nonempty list")
    validated = []
    seen_ids = set()
    for criterion in criteria:
        criterion = _mapping(criterion, "Criterion")
        criterion_id = _text(criterion, "criterion_id")
        if criterion_id in seen_ids:
            raise ValueError("Criterion IDs must be unique")
        seen_ids.add(criterion_id)
        validated.append((criterion_id, _text(criterion, "description"), _flag(criterion, "passed")))
    judge_passed = _flag(row, "judge_passed")
    if passed != (judge_passed and all(item[2] for item in validated)):
        raise ValueError("Response passed flag disagrees with the grade and criteria checks")
    failures = tuple(
        _attribution(
            case_id, "response", "response_generation",
            f"Failure observed in response criterion {criterion_id}: {description}",
        ) for criterion_id, description, criterion_passed in sorted(validated) if not criterion_passed
    )
    if not passed and not failures:
        # The existing scorer also honors an overall failed grade with all criteria passing.
        return (_attribution(
            case_id, "response", "response_generation",
            "Failure observed in the overall response grade; no individual criterion was marked failed.",
        ),)
    return failures


def attribute_report_failures(report: Mapping[str, Any], *, eval_type: EvalType) -> tuple[FailureAttribution, ...]:
    """Project one complete report onto safe fields; never inspect diagnostic payloads.

    Results are ordered by normalized case ID, then fixed check order (or criterion
    ID). Only results and the common total/passed/failed counters are required;
    redundant failures, metrics, and all content-bearing fields are left untouched.
    """
    functions = {
        "router": attribute_router_result, "specialist": attribute_specialist_result,
        "response": attribute_response_result, "conversation": attribute_conversation_result,
    }
    if not isinstance(eval_type, str) or eval_type not in functions:
        raise ValueError("Unsupported eval_type; choose router, specialist, response, or conversation")
    report = _mapping(report, "Report")
    results = report.get("results")
    if not isinstance(results, list):
        raise TypeError("results must be a list")
    for name in ("total", "passed", "failed"):
        if type(report.get(name)) is not int or report[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    rows = [_result(row) for row in results]
    ids = [case_id for _, case_id, _ in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Result case IDs must be unique")
    if (
        report["total"] != len(rows) or report["passed"] != sum(passed for _, _, passed in rows)
        or report["failed"] != len(rows) - report["passed"]
    ):
        raise ValueError("Report counts disagree with results")
    return tuple(
        attribution for row, _, _ in sorted(rows, key=lambda item: item[1])
        for attribution in functions[eval_type](row)
    )


def summarize_failure_attributions(attributions: Iterable[FailureAttribution]) -> dict[str, Any]:
    items = tuple(attributions)
    if not all(isinstance(item, FailureAttribution) for item in items):
        raise TypeError("Expected FailureAttribution objects")
    counts = {component: sum(item.component == component for item in items) for component in COMPONENTS}
    return {"total_attributions": len(items), "by_component": {name: count for name, count in counts.items() if count}}


def format_failure_attribution_report(attributions: Iterable[FailureAttribution]) -> str:
    """Display only component counts and sorted case IDs, never descriptions or payloads."""
    items = tuple(attributions)
    summary = summarize_failure_attributions(items)
    lines = ["Failure localization", f"Total attributions: {summary['total_attributions']}"]
    for component, count in summary["by_component"].items():
        case_ids = sorted({item.case_id for item in items if item.component == component})
        lines.append(f"{component}: {count} (cases: {', '.join(case_ids)})")
    if not items:
        lines.append("- none")
    return "\n".join(lines)
