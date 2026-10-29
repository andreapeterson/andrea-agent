"""Test final-response evaluation machinery with predetermined text and judgments."""

import json
import socket
from collections import Counter
from copy import deepcopy
from importlib import import_module
from unittest.mock import AsyncMock, call

import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from app.agents.prompt_context import PromptContext
from app.integrations import HandoffClient, LegacyCRMClient, SchedulerClient
from app.models.appointment import AppointmentSlot
from app.models.retrieval import PolicySearchResult
from app.services.policy_retriever import PolicyRetriever
from evals.response_eval import (
    DEFAULT_CASES_PATH,
    SPECIALIST_TOOLS,
    ResponseCriterionResult,
    ResponseEvalCase,
    ResponseJudgeResult,
    format_response_report,
    load_response_cases,
    run_response_evaluation,
)

FAKE_RESPONSES = {
    "verification": "I couldn't find an account with that phone number. Could you check the number?",
    "policy": "The clinic is open Monday through Friday, 8 a.m. to 6 p.m. Eastern Time.",
    "intake": "Is Maize experiencing uncontrolled bleeding?",
    "scheduling": "For Maize on November 3, 2026, I have 9:00 or 10:00 a.m. UTC-05:00 available. These are options, not a booking. Which do you prefer?",
    "human_handoff": "Your request for human assistance has been queued.",
}


def fake_grade(case, *, failed_ids=(), passed=True, reason="All supplied criteria are satisfied by this fixture."):
    return ResponseJudgeResult(
        passed=passed, reason=reason,
        criteria_results=[
            ResponseCriterionResult(criterion_id=criterion.criterion_id, passed=criterion.criterion_id not in failed_ids)
            for criterion in case.criteria
        ],
    )


@pytest.fixture(autouse=True)
def forbid_openai_and_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline response evaluation tests must not construct OpenAI clients or use the network")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(AsyncOpenAI, "__init__", forbidden)
    monkeypatch.setattr(OpenAI, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def valid_row():
    return json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8").splitlines()[0])


def write_rows(path, *rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_five_cases_cover_each_specialist_with_controlled_results_and_human_criteria() -> None:
    cases = load_response_cases()

    assert len(cases) == len({case.case_id for case in cases}) == 5
    assert Counter(case.specialist for case in cases) == {specialist: 1 for specialist in SPECIALIST_TOOLS}
    for case in cases:
        assert isinstance(case, ResponseEvalCase)
        assert isinstance(case.context, PromptContext)
        assert case.context.active_agent == case.specialist
        assert case.tool_name in SPECIALIST_TOOLS[case.specialist]
        assert len(case.criteria) == len({criterion.criterion_id for criterion in case.criteria}) == 3
        assert all(criterion.description for criterion in case.criteria)
        assert case.tags
    by_specialist = {case.specialist: case for case in cases}
    assert by_specialist["verification"].tool_result == {"found": False}
    policy = by_specialist["policy"].tool_result
    assert policy["found"] is True
    evidence = PolicySearchResult.model_validate(policy["evidence"][0])
    assert evidence.chunk.chunk_id == "demo-hours-1"
    assert "8:00 a.m. to 6:00 p.m. Eastern Time" in evidence.chunk.text
    intake = by_specialist["intake"].tool_result
    assert intake["routing_level"] == "needs_more_information"
    assert "difficulty_breathing" not in intake["missing_fields"]
    slots = by_specialist["scheduling"].tool_result["slots"]
    assert len(slots) == 2
    assert all(AppointmentSlot.model_validate(slot).starts_at.utcoffset() is not None for slot in slots)
    assert by_specialist["human_handoff"].tool_result == {"queued": True, "case_id": "demo-handoff-response-1"}


def test_loader_skips_blanks_and_normalizes_metadata(tmp_path, valid_row) -> None:
    valid_row["case_id"] = " trimmed-case "
    valid_row["criteria"][0]["criterion_id"] = " criterion-1 "
    valid_row["tags"] = [" tag "]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n \t\n" + json.dumps(valid_row) + "\n\n", encoding="utf-8")

    case, = load_response_cases(path)

    assert case.case_id == "trimmed-case"
    assert case.criteria[0].criterion_id == "criterion-1"
    assert case.tags == ["tag"]


@pytest.mark.parametrize("content", ["{broken", "[]", '{"tool_result":{"value":NaN}}'])
def test_loader_errors_include_filename_and_line_number(tmp_path, content) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + content, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_response_cases(path)
    assert str(caught.value).startswith(f"{path}:2:")


def test_duplicate_normalized_case_ids_are_rejected(tmp_path, valid_row) -> None:
    duplicate = {**valid_row, "case_id": " " + valid_row["case_id"] + " "}
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_response_cases(write_rows(tmp_path / "cases.jsonl", valid_row, duplicate))


@pytest.mark.parametrize("field, value", [
    ("specialist", "unsupported"), ("context", {"customer_id": "not-allowlisted"}),
    ("tool_name", "search_policy"), ("tool_result", None),
    ("case_id", " "), ("user_message", 5), ("criteria", []), ("tags", [" "]),
    ("criteria", [{"criterion_id": "id", "description": " "}]),
])
def test_pydantic_rejects_invalid_cases_with_loader_location(tmp_path, valid_row, field, value) -> None:
    valid_row[field] = value
    with pytest.raises(ValueError) as caught:
        load_response_cases(write_rows(tmp_path / "cases.jsonl", valid_row))
    assert str(caught.value).startswith(f"{tmp_path / 'cases.jsonl'}:1:")
    assert isinstance(caught.value.__cause__, ValidationError)


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate_criterion"])
def test_case_requires_fields_forbids_extras_and_duplicate_criteria(valid_row, change) -> None:
    if change == "missing":
        valid_row.pop("criteria")
    elif change == "extra":
        valid_row["extra"] = True
    else:
        valid_row["criteria"].append(valid_row["criteria"][0])
    with pytest.raises(ValidationError):
        ResponseEvalCase.model_validate(valid_row)


def test_null_tool_pair_is_supported_and_case_is_frozen(valid_row) -> None:
    valid_row.update(tool_name=None, tool_result=None)
    case = ResponseEvalCase.model_validate(valid_row)
    assert case.tool_name is case.tool_result is None
    with pytest.raises(ValidationError, match="frozen"):
        case.case_id = "replacement"
    valid_row["tool_result"] = {"found": False}
    with pytest.raises(ValidationError, match="both be null or both be supplied"):
        ResponseEvalCase.model_validate(valid_row)


@pytest.mark.asyncio
async def test_passing_responses_are_counted_and_each_callable_gets_the_controlled_case_once() -> None:
    cases = load_response_cases()
    before = deepcopy(cases)
    responses = [FAKE_RESPONSES[case.specialist] for case in cases]
    generate = AsyncMock(side_effect=responses)
    grade = AsyncMock(side_effect=[fake_grade(case) for case in cases])

    report = await run_response_evaluation(iter(cases), generate, grade)

    assert generate.await_args_list == [call(case) for case in cases]
    assert grade.await_args_list == [call(case, response) for case, response in zip(cases, responses, strict=True)]
    assert report["total"] == report["passed"] == 5
    assert report["failed"] == 0
    assert report["pass_rate"] == 1.0
    assert all(group == {"total": 1, "passed": 1, "failed": 0, "pass_rate": 1.0} for group in report["by_specialist"].values())
    assert report["failures"] == []
    assert [result["response"] for result in report["results"]] == responses
    assert all(not result["failed_criteria"] for result in report["results"])
    assert format_response_report(report).endswith("Failures:\n- none")
    assert cases == before
    json.dumps(report, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_passed", [True, False])
async def test_failed_criterion_fails_complete_case_and_appears_in_report(judge_passed) -> None:
    cases = load_response_cases()
    handoff = cases[-1]
    grades = [fake_grade(case) for case in cases[:-1]] + [fake_grade(
        handoff, failed_ids=["no_live_transfer_claim"], passed=judge_passed,
        reason="The response falsely claims a live connection to staff.",
    )]
    generate = AsyncMock(side_effect=[FAKE_RESPONSES[case.specialist] for case in cases[:-1]] + ["A staff member is now connected."])

    report = await run_response_evaluation(cases, generate, AsyncMock(side_effect=grades))

    assert (report["total"], report["passed"], report["failed"], report["pass_rate"]) == (5, 4, 1, 0.8)
    failure, = report["failures"]
    assert failure["case_id"] == "handoff_queued_not_connected"
    assert failure["failed_criteria"] == ["no_live_transfer_claim"]
    assert failure["judge_passed"] is judge_passed
    assert report["by_specialist"]["human_handoff"] == {"total": 1, "passed": 0, "failed": 1, "pass_rate": 0.0}
    text = format_response_report(report)
    assert "Total: 5\nPassed: 4\nFailed: 1\nPass rate: 80.0%" in text
    assert "human_handoff: 0/1 (0.0%)" in text
    assert "handoff_queued_not_connected: The response falsely claims a live connection to staff." in text
    assert "no_live_transfer_claim: Do not claim a live transfer" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["", " \t\n ", None, {"text": "Not a string"}])
async def test_blank_or_nonstring_responses_are_rejected_before_judging(response) -> None:
    judge = AsyncMock()
    with pytest.raises((ValueError, TypeError), match="generate_response must return"):
        await run_response_evaluation(load_response_cases()[:1], AsyncMock(return_value=response), judge)
    judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_judge_errors_propagate_instead_of_becoming_model_failures() -> None:
    error = RuntimeError("Judge failed")
    with pytest.raises(RuntimeError) as caught:
        await run_response_evaluation(load_response_cases()[:1], AsyncMock(return_value="Caller-facing text."), AsyncMock(side_effect=error))
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unknown"])
async def test_judge_must_grade_every_human_criterion_exactly_once(invalid) -> None:
    case = load_response_cases()[0]
    checks = list(fake_grade(case).criteria_results)
    if invalid == "missing":
        checks.pop()
    elif invalid == "duplicate":
        checks[-1] = checks[0]
    else:
        checks[-1] = ResponseCriterionResult(criterion_id="unknown", passed=True)
    grade = ResponseJudgeResult(passed=True, reason="Incomplete fixture judgment", criteria_results=checks)
    with pytest.raises(ValueError, match="each case criterion exactly once"):
        await run_response_evaluation([case], AsyncMock(return_value="Text"), AsyncMock(return_value=grade))


@pytest.mark.asyncio
async def test_unstructured_judge_results_are_rejected() -> None:
    with pytest.raises(TypeError, match="grade_response must return ResponseJudgeResult"):
        await run_response_evaluation(load_response_cases()[:1], AsyncMock(return_value="Text"), AsyncMock(return_value={"passed": True}))


@pytest.mark.parametrize("update", [
    {"passed": "true"}, {"reason": " "}, {"criteria_results": []}, {"extra": "forbidden"},
    {"criteria_results": [{"criterion_id": " ", "passed": True}]},
    {"criteria_results": [{"criterion_id": "id", "passed": 1}]},
])
def test_structured_judge_results_require_booleans_reason_and_criterion_results(update) -> None:
    payload = {"passed": True, "reason": "Fixture judgment", "criteria_results": [{"criterion_id": "id", "passed": True}], **update}
    with pytest.raises(ValidationError):
        ResponseJudgeResult.model_validate(payload)


@pytest.mark.asyncio
async def test_empty_input_calls_nothing_and_has_safe_rates() -> None:
    generate, grade = AsyncMock(), AsyncMock()
    report = await run_response_evaluation([], generate, grade)
    generate.assert_not_awaited()
    grade.assert_not_awaited()
    assert report["total"] == report["passed"] == report["failed"] == 0
    assert report["pass_rate"] == 0.0
    assert all(group["pass_rate"] == 0.0 for group in report["by_specialist"].values())
    assert "Pass rate: N/A (0 cases)" in format_response_report(report)


@pytest.mark.asyncio
async def test_no_production_agents_handlers_clients_or_tools_are_used(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Response evaluation must use only injected generators, judges, and controlled results")

    for cls in (LegacyCRMClient, SchedulerClient, HandoffClient, PolicyRetriever):
        monkeypatch.setattr(cls, "__init__", forbidden)
    for cls, methods in (
        (LegacyCRMClient, ["find_customer_by_phone"]), (SchedulerClient, ["find_slots", "book_appointment"]),
        (HandoffClient, ["create_handoff"]), (PolicyRetriever, ["search"]),
    ):
        for method in methods:
            monkeypatch.setattr(cls, method, forbidden)
    factories = {
        "lookup_customer_tool": ["create_lookup_customer_handler"],
        "search_policy_tool": ["create_search_policy_handler"],
        "record_intake_tool": ["create_record_intake_handler"],
        "request_handoff_tool": ["create_request_handoff_handler"],
        "scheduling_tools": ["create_find_appointment_slots_handler", "create_select_appointment_slot_handler", "create_book_selected_appointment_handler"],
    }
    for module_name, names in factories.items():
        for name in names:
            monkeypatch.setattr(import_module("app.agents." + module_name), name, forbidden)
    for module_name, class_name in (
        ("verification_agent", "VerificationAgent"), ("policy_agent", "PolicyAgent"),
        ("intake_agent", "IntakeAgent"), ("scheduling_agent", "SchedulingAgent"),
        ("human_handoff_agent", "HumanHandoffAgent"),
    ):
        module = import_module("app.agents." + module_name)
        monkeypatch.setattr(getattr(module, class_name), "__init__", forbidden)
        monkeypatch.setattr(module, "run_agent_turn", forbidden)
    monkeypatch.setattr(import_module("app.agents.responses_tool_loop"), "run_agent_turn", forbidden)
    cases = load_response_cases()

    report = await run_response_evaluation(
        cases, AsyncMock(side_effect=[FAKE_RESPONSES[case.specialist] for case in cases]),
        AsyncMock(side_effect=[fake_grade(case) for case in cases]),
    )

    assert report["passed"] == 5
