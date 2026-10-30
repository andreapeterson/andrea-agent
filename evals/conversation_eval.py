"""Score multi-turn workflow observations from an injected offline runner."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.conversation import ConversationPhase
from evals.specialist_eval import SPECIALIST_TOOLS

DEFAULT_CASES_PATH = Path(__file__).with_name("conversation_cases.jsonl")
Specialist = Literal["verification", "policy", "intake", "scheduling", "human_handoff"]
NonblankString = Annotated[str, Field(min_length=1)]
CHECK_NAMES = (
    "specialist_sequence", "tool_sequence", "final_phase", "verified_customer",
    "booking_confirmed", "handoff_queued", "complete_case",
)


def _validate_turns(specialists: list[str], tools: list[list[str]]) -> None:
    if len(specialists) != len(tools):
        raise ValueError("specialist and tool lists must have the same number of turns")
    for turn, (specialist, names) in enumerate(zip(specialists, tools, strict=True), start=1):
        for name in names:
            if name not in SPECIALIST_TOOLS[specialist]:
                raise ValueError(f"turn {turn}: tool {name!r} does not belong to specialist {specialist!r}")


def _validate_outcomes(phase: ConversationPhase, booking: bool, handoff: bool) -> None:
    if booking and handoff:
        raise ValueError("booking_confirmed and handoff_queued cannot both be true")
    if phase == ConversationPhase.BOOKING_COMPLETE and not booking:
        raise ValueError("booking_complete requires booking_confirmed")
    if phase == ConversationPhase.HANDOFF_COMPLETE and not handoff:
        raise ValueError("handoff_complete requires handoff_queued")


class ConversationEvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    caller_turns: list[NonblankString] = Field(min_length=1)
    expected_specialists_by_turn: list[Specialist] = Field(min_length=1)
    expected_tools_by_turn: list[list[NonblankString]] = Field(min_length=1)
    expected_final_phase: ConversationPhase
    expected_verified_customer: bool = Field(strict=True)
    expected_booking_confirmed: bool = Field(strict=True)
    expected_handoff_queued: bool = Field(strict=True)
    tags: list[NonblankString] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_workflow(self) -> Self:
        _validate_turns(self.expected_specialists_by_turn, self.expected_tools_by_turn)
        if len(self.expected_specialists_by_turn) != len(self.caller_turns):
            raise ValueError("expectation lists must have the same length as caller_turns")
        _validate_outcomes(
            self.expected_final_phase, self.expected_booking_confirmed, self.expected_handoff_queued,
        )
        return self


class ConversationRunObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    specialists_by_turn: list[Specialist] = Field(min_length=1)
    tools_by_turn: list[list[NonblankString]] = Field(min_length=1)
    final_phase: ConversationPhase
    verified_customer: bool = Field(strict=True)
    booking_confirmed: bool = Field(strict=True)
    handoff_queued: bool = Field(strict=True)

    @model_validator(mode="after")
    def validate_workflow(self) -> Self:
        _validate_turns(self.specialists_by_turn, self.tools_by_turn)
        _validate_outcomes(self.final_phase, self.booking_confirmed, self.handoff_queued)
        return self


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def load_conversation_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[ConversationEvalCase]:
    """Read golden workflows line by line, retaining file locations in errors."""
    cases = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                case = ConversationEvalCase.model_validate(json.loads(line, parse_constant=_reject_json_constant))
                if case.case_id in seen_ids:
                    raise ValueError(f"duplicate case_id: {case.case_id}")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            seen_ids.add(case.case_id)
            cases.append(case)
    return cases


async def run_conversation_evaluation(
    cases: Iterable[ConversationEvalCase],
    run_conversation: Callable[[ConversationEvalCase], Awaitable[ConversationRunObservation]],
) -> dict[str, Any]:
    """Compare exact sequences and outcomes; runner and validation errors propagate."""
    cases = list(cases)
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("conversation case IDs must be unique")
    results = []
    for case in cases:
        observation = await run_conversation(case)
        if not isinstance(observation, ConversationRunObservation):
            raise TypeError("run_conversation must return ConversationRunObservation")
        if (
            len(observation.specialists_by_turn) != len(case.caller_turns)
            or len(observation.tools_by_turn) != len(case.caller_turns)
        ):
            raise ValueError(f"{case.case_id}: observed turn lists must have the same length as caller_turns")
        expected = {
            "specialist_sequence": case.expected_specialists_by_turn,
            "tool_sequence": case.expected_tools_by_turn,
            "final_phase": case.expected_final_phase.value,
            "verified_customer": case.expected_verified_customer,
            "booking_confirmed": case.expected_booking_confirmed,
            "handoff_queued": case.expected_handoff_queued,
        }
        actual = {
            "specialist_sequence": observation.specialists_by_turn,
            "tool_sequence": observation.tools_by_turn,
            "final_phase": observation.final_phase.value,
            "verified_customer": observation.verified_customer,
            "booking_confirmed": observation.booking_confirmed,
            "handoff_queued": observation.handoff_queued,
        }
        checks = {name: actual[name] == value for name, value in expected.items()}
        checks["complete_case"] = all(checks.values())
        results.append({
            "case_id": case.case_id,
            "expected": expected,
            "actual": actual,
            "checks": checks,
            "passed": checks["complete_case"],
            "failure_details": [
                f"{name}: expected {json.dumps(expected[name])}, received {json.dumps(actual[name])}"
                for name in expected if not checks[name]
            ],
            "tags": list(case.tags),
        })
    total = len(results)
    passed = sum(result["passed"] for result in results)
    metrics = {}
    for name in CHECK_NAMES:
        count = sum(result["checks"][name] for result in results)
        metrics[name] = {"passed": count, "total": total, "accuracy": count / total if total else 0.0}
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "metrics": metrics,
        "results": results,
        "failures": [result for result in results if not result["passed"]],
    }


def format_conversation_report(report: dict[str, Any]) -> str:
    """Show all checks and exact expected-versus-actual workflow mismatches."""
    rate = f"{report['pass_rate']:.1%}" if report["total"] else "N/A (0 applicable cases)"
    lines = [
        "Conversation evaluation", f"Total: {report['total']}",
        f"Passed: {report['passed']}", f"Failed: {report['failed']}", f"Pass rate: {rate}",
        "", "Scores:",
    ]
    for name, metric in report["metrics"].items():
        score = (
            f"{metric['passed']}/{metric['total']} ({metric['accuracy']:.1%})"
            if metric["total"] else "N/A (0 applicable cases)"
        )
        lines.append(f"{name}: {score}")
    lines.extend(["", "Failures:"])
    for failure in report["failures"]:
        lines.append(f"- {failure['case_id']}: {'; '.join(failure['failure_details'])}")
    if not report["failures"]:
        lines.append("- none")
    return "\n".join(lines)
