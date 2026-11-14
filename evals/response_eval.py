"""Grade final caller-facing text with injected generators and judges only."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.prompt_context import PromptContext
from evals.specialist_eval import TOOL_ARGUMENT_MODELS

DEFAULT_CASES_PATH = Path(__file__).with_name("response_cases.jsonl")
SPECIALIST_TOOLS = {
    "verification": {"lookup_customer"},
    "policy": {"search_policy"},
    "intake": {"record_intake", "request_handoff"},
    "scheduling": {"find_appointment_slots", "select_appointment_slot", "book_selected_appointment"},
    "human_handoff": {"request_handoff"},
}


class ResponseCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    criterion_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ResponseEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    specialist: Literal["verification", "policy", "intake", "scheduling", "human_handoff"]
    user_message: str = Field(min_length=1)
    context: PromptContext
    tool_name: str | None = Field(min_length=1)
    tool_arguments: dict[str, object] | None
    tool_result: dict[str, object] | None
    criteria: list[ResponseCriterion] = Field(min_length=1)
    tags: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        tool_fields = (self.tool_name, self.tool_arguments, self.tool_result)
        if any(value is None for value in tool_fields) and not all(value is None for value in tool_fields):
            raise ValueError("tool_name, tool_arguments, and tool_result must all be null or all be supplied")
        if self.tool_name is not None and self.tool_name not in SPECIALIST_TOOLS[self.specialist]:
            raise ValueError("tool_name must belong to the selected specialist")
        if self.tool_name is not None:
            TOOL_ARGUMENT_MODELS[self.tool_name].model_validate(self.tool_arguments)
        ids = [criterion.criterion_id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique within a case")
        json.dumps([self.tool_arguments, self.tool_result], allow_nan=False)
        return self


class ResponseCriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True, strict=True)

    criterion_id: str = Field(min_length=1)
    passed: bool


class ResponseJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True, strict=True)

    passed: bool
    reason: str = Field(min_length=1)
    criteria_results: list[ResponseCriterionResult] = Field(min_length=1)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def load_response_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[ResponseEvalCase]:
    """Read controlled fixtures, reporting invalid rows with filename and line number."""
    cases = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_json_constant)
                case = ResponseEvalCase.model_validate(row)
                if case.case_id in seen_ids:
                    raise ValueError(f"duplicate case_id: {case.case_id}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen_ids.add(case.case_id)
            cases.append(case)
    return cases


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(result["passed"] for result in results)
    return {"total": total, "passed": passed, "failed": total - passed, "pass_rate": passed / total if total else 0.0}


async def run_response_evaluation(
    cases: Iterable[ResponseEvalCase],
    generate_response: Callable[[ResponseEvalCase], Awaitable[str]],
    grade_response: Callable[[ResponseEvalCase, str], Awaitable[ResponseJudgeResult]],
) -> dict[str, Any]:
    """Evaluate cases sequentially without constructing agents, clients, or handlers."""
    results = []
    for case in cases:
        response = await generate_response(case)
        if not isinstance(response, str):
            raise TypeError("generate_response must return a string")
        if not response.strip():
            raise ValueError("generate_response must return nonblank caller-facing text")
        grade = await grade_response(case, response)
        if not isinstance(grade, ResponseJudgeResult):
            raise TypeError("grade_response must return ResponseJudgeResult")
        judged_ids = [result.criterion_id for result in grade.criteria_results]
        expected_ids = {criterion.criterion_id for criterion in case.criteria}
        if len(judged_ids) != len(expected_ids) or set(judged_ids) != expected_ids:
            raise ValueError("grade_response must grade each case criterion exactly once")
        checks = {result.criterion_id: result.passed for result in grade.criteria_results}
        results.append({
            "case_id": case.case_id,
            "specialist": case.specialist,
            "response": response,
            "passed": grade.passed and all(checks.values()),
            "judge_passed": grade.passed,
            "reason": grade.reason,
            "criteria_results": [
                {**criterion.model_dump(), "passed": checks[criterion.criterion_id]}
                for criterion in case.criteria
            ],
            "failed_criteria": [criterion.criterion_id for criterion in case.criteria if not checks[criterion.criterion_id]],
            "tags": list(case.tags),
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


def format_response_report(report: dict[str, Any]) -> str:
    """Display counts, pass rates, judge reasons, and failed human-written criteria."""
    rate = f"{report['pass_rate']:.1%}" if report["total"] else "N/A (0 cases)"
    lines = [
        "Specialist response evaluation", f"Total: {report['total']}",
        f"Passed: {report['passed']}", f"Failed: {report['failed']}", f"Pass rate: {rate}",
        "", "By specialist:",
    ]
    for specialist, group in report["by_specialist"].items():
        rate = f"{group['pass_rate']:.1%}" if group["total"] else "N/A"
        lines.append(f"{specialist}: {group['passed']}/{group['total']} ({rate})")
    lines.extend(["", "Failures:"])
    for failure in report["failures"]:
        lines.append(f"- {failure['case_id']}: {failure['reason']}")
        for criterion in failure["criteria_results"]:
            if not criterion["passed"]:
                lines.append(f"  - {criterion['criterion_id']}: {criterion['description']}")
    if not report["failures"]:
        lines.append("- none")
    return "\n".join(lines)
