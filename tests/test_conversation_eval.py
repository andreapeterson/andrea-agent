"""Exercise workflow loading and scoring with injected observations only."""

import json
import socket
from copy import deepcopy
from importlib import import_module
from unittest.mock import AsyncMock, call

import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from app.integrations import HandoffClient, LegacyCRMClient, SchedulerClient
from app.models.conversation import ConversationPhase, ConversationState
from app.services.conversation_store import InMemoryConversationStore
from app.services.policy_retriever import PolicyRetriever
from app.services.prompt_agent_runtime import PromptAgentRuntime
from evals.conversation_eval import (
    CHECK_NAMES,
    DEFAULT_CASES_PATH,
    ConversationEvalCase,
    ConversationRunObservation,
    format_conversation_report,
    load_conversation_cases,
    run_conversation_evaluation,
)


@pytest.fixture(autouse=True)
def forbid_production_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline conversation evaluation must use only injected observations")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for cls in (
        AsyncOpenAI, OpenAI, PromptAgentRuntime, InMemoryConversationStore, ConversationState,
        LegacyCRMClient, SchedulerClient, HandoffClient, PolicyRetriever,
    ):
        monkeypatch.setattr(cls, "__init__", forbidden)
    for cls, methods in (
        (PromptAgentRuntime, ["start_conversation", "handle_turn"]),
        (InMemoryConversationStore, ["create", "get", "save"]),
        (LegacyCRMClient, ["find_customer_by_phone"]),
        (SchedulerClient, ["find_slots", "book_appointment"]),
        (HandoffClient, ["create_handoff"]), (PolicyRetriever, ["search"]),
    ):
        for method in methods:
            monkeypatch.setattr(cls, method, forbidden)
    for module_name, names in {
        "lookup_customer_tool": ["create_lookup_customer_handler"],
        "record_intake_tool": ["create_record_intake_handler"],
        "request_handoff_tool": ["create_request_handoff_handler"],
        "search_policy_tool": ["create_search_policy_handler"],
        "scheduling_tools": [
            "create_find_appointment_slots_handler", "create_select_appointment_slot_handler",
            "create_book_selected_appointment_handler",
        ],
    }.items():
        for name in names:
            monkeypatch.setattr(import_module("app.agents." + module_name), name, forbidden)
    for module_name, class_name in (
        ("verification_agent", "VerificationAgent"), ("policy_agent", "PolicyAgent"),
        ("intake_agent", "IntakeAgent"), ("scheduling_agent", "SchedulingAgent"),
        ("human_handoff_agent", "HumanHandoffAgent"), ("router_agent", "OpenAIRouter"),
    ):
        monkeypatch.setattr(getattr(import_module("app.agents." + module_name), class_name), "__init__", forbidden)
    monkeypatch.setattr(import_module("app.agents.responses_tool_loop"), "run_agent_turn", forbidden)


def observation_payload(case):
    return {
        "specialists_by_turn": list(case.expected_specialists_by_turn),
        "tools_by_turn": deepcopy(case.expected_tools_by_turn),
        "final_phase": case.expected_final_phase.value,
        "verified_customer": case.expected_verified_customer,
        "booking_confirmed": case.expected_booking_confirmed,
        "handoff_queued": case.expected_handoff_queued,
    }


@pytest.fixture
def valid_row():
    return json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8").splitlines()[0])


def write_rows(path, *rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_three_golden_workflows_have_the_requested_structure():
    policy, urgent, routine = load_conversation_cases()

    assert len({case.case_id for case in (policy, urgent, routine)}) == 3
    assert all(isinstance(case, ConversationEvalCase) for case in (policy, urgent, routine))
    assert policy.expected_specialists_by_turn == ["policy"]
    assert policy.expected_tools_by_turn == [["search_policy"]]
    assert not policy.expected_verified_customer
    assert not policy.expected_booking_confirmed
    assert not policy.expected_handoff_queued
    assert urgent.expected_specialists_by_turn == ["verification", "intake"]
    assert urgent.expected_tools_by_turn == [["lookup_customer"], ["record_intake", "request_handoff"]]
    assert "Maize" in urgent.caller_turns[1]
    assert "difficulty breathing" in urgent.caller_turns[1]
    assert urgent.expected_final_phase == ConversationPhase.HANDOFF_COMPLETE
    assert urgent.expected_verified_customer and urgent.expected_handoff_queued
    assert not urgent.expected_booking_confirmed
    assert routine.expected_specialists_by_turn == ["verification", "intake", "scheduling", "scheduling", "scheduling"]
    assert routine.expected_tools_by_turn == [
        ["lookup_customer"], ["record_intake"], ["find_appointment_slots"],
        ["select_appointment_slot"], ["book_selected_appointment"],
    ]
    assert "demo-routine-slot-1" in routine.caller_turns[3]
    assert routine.caller_turns[-1] == "Yes, please book that appointment."
    assert routine.expected_final_phase == ConversationPhase.BOOKING_COMPLETE
    assert routine.expected_verified_customer and routine.expected_booking_confirmed
    assert not routine.expected_handoff_queued


def test_loader_skips_blank_lines_and_strips_metadata(tmp_path, valid_row):
    valid_row.update(case_id=" trimmed ", tags=[" tag "], caller_turns=["  Hours?  "])
    path = tmp_path / "cases.jsonl"
    path.write_text("\n \t\n" + json.dumps(valid_row) + "\n\n", encoding="utf-8")
    case, = load_conversation_cases(path)
    assert (case.case_id, case.tags, case.caller_turns) == ("trimmed", ["tag"], ["Hours?"])


def test_duplicate_normalized_ids_fail_with_location(tmp_path, valid_row):
    duplicate = {**valid_row, "case_id": " " + valid_row["case_id"] + " "}
    path = write_rows(tmp_path / "cases.jsonl", valid_row, duplicate)
    with pytest.raises(ValueError) as caught:
        load_conversation_cases(path)
    assert str(caught.value).startswith(f"{path}:2:")
    assert "duplicate case_id" in str(caught.value)


@pytest.mark.parametrize("content", ["{broken", "[]", "null", '{"case_id":NaN}'])
def test_invalid_json_or_rows_fail_with_location(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + content, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_conversation_cases(path)
    assert str(caught.value).startswith(f"{path}:2:")


@pytest.mark.parametrize("update", [
    {"case_id": " "}, {"caller_turns": []}, {"caller_turns": [" "]},
    {"caller_turns": [7]}, {"tags": []}, {"tags": [" "]}, {"tags": "policy"},
    {"expected_specialists_by_turn": ["router"]},
    {"expected_specialists_by_turn": ["booking"]},
    {"expected_tools_by_turn": [["lookup_customer"]]},
    {"expected_tools_by_turn": [["unknown_tool"]]},
    {"expected_tools_by_turn": [[" "]]}, {"expected_tools_by_turn": ["search_policy"]},
    {"expected_specialists_by_turn": ["policy", "policy"]},
    {"expected_tools_by_turn": [["search_policy"], []]},
    {"caller_turns": ["Hours?", "Prices?"]},
    {"expected_final_phase": "invented"}, {"expected_verified_customer": "false"},
    {"expected_booking_confirmed": 0}, {"expected_handoff_queued": "true"}, {"extra": True},
])
def test_invalid_expectations_fail_pydantic_and_loader_validation(tmp_path, valid_row, update):
    valid_row.update(update)
    with pytest.raises(ValidationError):
        ConversationEvalCase.model_validate(valid_row)
    path = write_rows(tmp_path / "cases.jsonl", valid_row)
    with pytest.raises(ValueError) as caught:
        load_conversation_cases(path)
    assert str(caught.value).startswith(f"{path}:1:")
    assert isinstance(caught.value.__cause__, ValidationError)


@pytest.mark.parametrize("model", [ConversationEvalCase, ConversationRunObservation])
def test_models_require_all_fields_and_are_frozen(model, valid_row):
    payload = valid_row if model is ConversationEvalCase else observation_payload(load_conversation_cases()[0])
    for field in model.model_fields:
        with pytest.raises(ValidationError, match=field):
            model.model_validate({key: value for key, value in payload.items() if key != field})
    instance = model.model_validate(payload)
    field = "case_id" if model is ConversationEvalCase else "verified_customer"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(instance, field, "replacement")


@pytest.mark.parametrize("update", [
    {"specialists_by_turn": ["router"]}, {"tools_by_turn": [["lookup_customer"]]},
    {"tools_by_turn": [["unknown_tool"]]}, {"tools_by_turn": [[" "]]},
    {"specialists_by_turn": ["policy", "policy"]}, {"tools_by_turn": []},
    {"final_phase": "invented"}, {"verified_customer": "false"},
    {"booking_confirmed": 0}, {"handoff_queued": "false"}, {"extra": True},
])
def test_observation_validates_turn_lengths_names_types_and_ownership(update):
    payload = observation_payload(load_conversation_cases()[0])
    with pytest.raises(ValidationError):
        ConversationRunObservation.model_validate({**payload, **update})


@pytest.mark.parametrize("model", [ConversationEvalCase, ConversationRunObservation])
@pytest.mark.parametrize("phase, booking, handoff, error", [
    ("started", True, True, "cannot both be true"),
    ("booking_complete", False, False, "booking_complete requires booking_confirmed"),
    ("handoff_complete", False, False, "handoff_complete requires handoff_queued"),
])
def test_final_outcome_consistency(model, phase, booking, handoff, error, valid_row):
    if model is ConversationEvalCase:
        payload = {**valid_row, "expected_final_phase": phase, "expected_booking_confirmed": booking, "expected_handoff_queued": handoff}
    else:
        payload = {**observation_payload(load_conversation_cases()[0]), "final_phase": phase, "booking_confirmed": booking, "handoff_queued": handoff}
    with pytest.raises(ValidationError, match=error):
        model.model_validate(payload)


@pytest.mark.asyncio
async def test_perfect_injected_observations_pass_sequentially_without_production_execution():
    cases = load_conversation_cases()
    before = deepcopy(cases)
    observed_cases = []

    async def run(case):
        observed_cases.append(case)
        return ConversationRunObservation.model_validate(observation_payload(case))

    report = await run_conversation_evaluation(iter(cases), run)

    assert observed_cases == cases == before
    assert report["total"] == report["passed"] == 3
    assert report["failed"] == 0
    assert report["pass_rate"] == 1.0
    assert set(report["metrics"]) == set(CHECK_NAMES)
    assert all(metric == {"passed": 3, "total": 3, "accuracy": 1.0} for metric in report["metrics"].values())
    assert [row["case_id"] for row in report["results"]] == [case.case_id for case in cases]
    assert report["failures"] == []
    assert all(all(row["checks"].values()) for row in report["results"])
    assert "complete_case: 3/3 (100.0%)" in format_conversation_report(report)
    assert "Failures:\n- none" in format_conversation_report(report)


@pytest.mark.asyncio
async def test_wrong_specialist_fails_only_specialist_sequence_and_complete_case(valid_row):
    # A speech turn permits either specialist without also changing tool ownership.
    case = ConversationEvalCase.model_validate({**valid_row, "expected_tools_by_turn": [[]]})
    payload = {**observation_payload(case), "specialists_by_turn": ["verification"]}
    runner = AsyncMock(return_value=ConversationRunObservation.model_validate(payload))
    report = await run_conversation_evaluation([case], runner)
    runner.assert_awaited_once_with(case)
    assert {name for name, passed in report["results"][0]["checks"].items() if not passed} == {"specialist_sequence", "complete_case"}
    assert report["results"][0]["failure_details"] == ['specialist_sequence: expected ["policy"], received ["verification"]']


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [["request_handoff", "record_intake"], ["record_intake"], ["record_intake", "request_handoff", "request_handoff"]])
async def test_exact_tool_order_missing_and_additional_calls_are_reported(tools):
    case = load_conversation_cases()[1]
    payload = observation_payload(case)
    payload["tools_by_turn"][1] = tools
    report = await run_conversation_evaluation([case], AsyncMock(return_value=ConversationRunObservation.model_validate(payload)))
    result = report["results"][0]
    assert {name for name, passed in result["checks"].items() if not passed} == {"tool_sequence", "complete_case"}
    text = format_conversation_report(report)
    assert case.case_id in text
    assert 'tool_sequence: expected [["lookup_customer"], ["record_intake", "request_handoff"]]' in text
    assert f"received {json.dumps(payload['tools_by_turn'])}" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("field, value", [
    ("final_phase", "completed"), ("verified_customer", True),
    ("booking_confirmed", True), ("handoff_queued", True),
])
async def test_phase_and_each_outcome_are_scored_independently(field, value):
    case = load_conversation_cases()[0]
    payload = {**observation_payload(case), field: value}
    report = await run_conversation_evaluation([case], AsyncMock(return_value=ConversationRunObservation.model_validate(payload)))
    assert {name for name, passed in report["results"][0]["checks"].items() if not passed} == {field, "complete_case"}
    assert report["metrics"][field] == {"passed": 0, "total": 1, "accuracy": 0.0}
    assert f"{field}: expected" in format_conversation_report(report)
    assert json.dumps(value) in format_conversation_report(report)


@pytest.mark.asyncio
async def test_wrong_phase_and_missing_outcomes_have_separate_failure_details():
    case = load_conversation_cases()[2]
    payload = {**observation_payload(case), "final_phase": "confirming_booking", "verified_customer": False, "booking_confirmed": False}
    report = await run_conversation_evaluation([case], AsyncMock(return_value=ConversationRunObservation.model_validate(payload)))
    result = report["results"][0]
    assert report["failed"] == 1 and report["pass_rate"] == 0.0
    assert len(result["failure_details"]) == 3
    assert 'final_phase: expected "booking_complete", received "confirming_booking"' in result["failure_details"]
    assert "verified_customer: expected true, received false" in result["failure_details"]
    assert "booking_confirmed: expected true, received false" in result["failure_details"]
    assert result["checks"]["tool_sequence"] and result["checks"]["specialist_sequence"]
    assert report["failures"] == [result]


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [1, 3])
async def test_observed_turn_count_must_match_case(turns):
    case = load_conversation_cases()[1]
    payload = {**observation_payload(case), "specialists_by_turn": ["policy"] * turns, "tools_by_turn": [[] for _ in range(turns)]}
    observation = ConversationRunObservation.model_validate(payload)
    with pytest.raises(ValueError, match=f"{case.case_id}: observed turn lists"):
        await run_conversation_evaluation([case], AsyncMock(return_value=observation))


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, {}, "handoff_complete", True])
async def test_wrong_runner_return_type_raises(value):
    with pytest.raises(TypeError, match="run_conversation must return ConversationRunObservation"):
        await run_conversation_evaluation(load_conversation_cases()[:1], AsyncMock(return_value=value))


@pytest.mark.asyncio
async def test_runner_errors_propagate_and_stop_evaluation():
    cases = load_conversation_cases()
    error = RuntimeError("injected runner failed")
    runner = AsyncMock(side_effect=error)
    with pytest.raises(RuntimeError) as caught:
        await run_conversation_evaluation(cases, runner)
    assert caught.value is error
    assert runner.await_args_list == [call(cases[0])]


@pytest.mark.asyncio
async def test_duplicate_in_memory_case_ids_fail_before_runner():
    case = load_conversation_cases()[0]
    runner = AsyncMock()
    with pytest.raises(ValueError, match="case IDs must be unique"):
        await run_conversation_evaluation([case, case], runner)
    runner.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_evaluation_has_zero_numeric_rates_and_na_terminal_metrics():
    runner = AsyncMock()
    report = await run_conversation_evaluation([], runner)
    runner.assert_not_awaited()
    assert report["total"] == report["passed"] == report["failed"] == 0
    assert report["pass_rate"] == 0.0
    assert report["results"] == report["failures"] == []
    assert all(metric == {"passed": 0, "total": 0, "accuracy": 0.0} for metric in report["metrics"].values())
    text = format_conversation_report(report)
    assert "Pass rate: N/A (0 applicable cases)" in text
    for name in CHECK_NAMES:
        assert f"{name}: N/A (0 applicable cases)" in text
