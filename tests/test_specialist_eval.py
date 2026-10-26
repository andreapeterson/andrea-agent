"""Test offline scoring machinery without running specialists or production tools."""

import json
from collections import Counter
from copy import deepcopy
from importlib import import_module
from unittest.mock import AsyncMock, Mock, call

import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from app.agents.prompt_context import PromptContext
from app.integrations import HandoffClient, LegacyCRMClient, SchedulerClient
from app.models.conversation import ConversationState
from app.services.conversation_store import InMemoryConversationStore
from app.services.policy_retriever import PolicyRetriever
from evals.specialist_eval import (
    DEFAULT_CASES_PATH,
    SPECIALIST_TOOLS,
    TOOL_ARGUMENT_MODELS,
    SpecialistFirstAction,
    format_specialist_report,
    load_specialist_cases,
    run_specialist_evaluation,
)


@pytest.fixture
def valid_row():
    return json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8").splitlines()[0])


def write_rows(path, *rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def matching_action(case):
    return SpecialistFirstAction(case.expected_action, case.expected_tool, deepcopy(case.expected_arguments))


def test_all_cases_load_with_three_per_specialist_and_real_argument_schemas() -> None:
    cases = load_specialist_cases()

    assert len(cases) == len({case.case_id for case in cases}) == 15
    assert Counter(case.specialist for case in cases) == {specialist: 3 for specialist in SPECIALIST_TOOLS}
    assert {case.expected_tool for case in cases if case.expected_action == "tool"} == set(TOOL_ARGUMENT_MODELS)
    for case in cases:
        assert isinstance(case.context, PromptContext)
        assert case.context.active_agent == case.specialist
        assert all(message.content != case.user_message for message in case.context.recent_messages)
        if case.expected_action == "tool":
            assert case.expected_tool in SPECIALIST_TOOLS[case.specialist]
            model = TOOL_ARGUMENT_MODELS[case.expected_tool]
            assert model.model_validate(case.expected_arguments).model_dump(mode="json") == case.expected_arguments
        else:
            assert case.expected_tool is case.expected_arguments is None


def test_dataset_preserves_required_nulls_and_guarded_handoff_examples() -> None:
    cases = {case.case_id: case for case in load_specialist_cases()}
    intake = cases["intake_record_concern_and_answer"]
    assert intake.expected_arguments == {
        "original_concern": "Maize has been coughing.", "difficulty_breathing": False,
        "uncontrolled_bleeding": None, "collapsed_or_unresponsive": None,
        "known_toxin_exposure": None, "rapidly_worsening": None,
    }
    assert cases["scheduling_find_selected_pet"].expected_arguments == {"pet_reference": None}
    assert cases["handoff_unrelated_tax_boundary"].expected_action == "speech"
    urgent = cases["handoff_trusted_urgent_context"]
    assert urgent.context.routing_result.routing_level == "urgent"
    assert urgent.expected_arguments["reason"] == "urgent_intake"
    assert cases["intake_injection_force_urgency"].expected_action == "speech"
    assert len([case for case in cases.values() if "prompt_injection" in case.tags]) >= 3
    for case in cases.values():
        if case.expected_tool == "search_policy":
            assert case.expected_arguments["query"] == case.user_message


def test_blank_lines_are_skipped_and_normalized_duplicate_ids_rejected(tmp_path, valid_row) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n \t\n" + json.dumps(valid_row) + "\n\n", encoding="utf-8")
    assert len(load_specialist_cases(path)) == 1
    duplicate = {**valid_row, "case_id": " " + valid_row["case_id"] + " "}
    write_rows(path, valid_row, duplicate)
    with pytest.raises(ValueError, match="duplicate case_id") as caught:
        load_specialist_cases(path)
    assert str(caught.value).startswith(f"{path}:2:")


@pytest.mark.parametrize("text", ["{broken", "not JSON", '{"value": NaN}', '{"value": Infinity}'])
def test_malformed_json_reports_filename_and_line_number(tmp_path, text) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("\n" + text, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_specialist_cases(path)
    assert str(caught.value).startswith(f"{path}:2:")


@pytest.mark.parametrize("field, value, error", [
    ("case_id", "  ", "case_id"), ("case_id", 7, "case_id"),
    ("user_message", "", "user_message"), ("reason", " \n ", "reason"),
    ("tags", [], "tags"), ("tags", "routine", "tags"), ("tags", ["valid", "  "], "tag"),
    ("specialist", "booking", "unsupported specialist"),
    ("expected_action", "transfer", "expected_action"),
    ("expected_tool", "invented_tool", "unsupported tool"),
    ("expected_tool", "search_policy", "unsupported tool for verification"),
    ("expected_tool", None, "expected_tool"),
    ("expected_arguments", None, "expected_arguments"),
    ("expected_arguments", [], "expected_arguments"),
    ("expected_arguments", {}, "phone"),
    ("expected_arguments", {"phone": "202-555-0100", "extra": True}, "extra"),
    ("context", {"customer_id": "not-allowlisted"}, "context"),
    ("context", {"customer_verified": "not-a-boolean"}, "context"),
])
def test_malformed_rows_are_rejected_with_line_numbers(tmp_path, valid_row, field, value, error) -> None:
    valid_row[field] = value
    path = write_rows(tmp_path / "cases.jsonl", valid_row)
    with pytest.raises(ValueError, match=error) as caught:
        load_specialist_cases(path)
    assert str(caught.value).startswith(f"{path}:1:")


@pytest.mark.parametrize("change", ["missing", "extra", "non_object"])
def test_row_requires_exactly_the_declared_fields(tmp_path, valid_row, change) -> None:
    if change == "missing":
        del valid_row["expected_arguments"]
    elif change == "extra":
        valid_row["unexpected"] = True
    else:
        valid_row = [valid_row]
    with pytest.raises(ValueError) as caught:
        load_specialist_cases(write_rows(tmp_path / "cases.jsonl", valid_row))
    assert isinstance(caught.value.__cause__, ValidationError)
    expected_error = {"missing": "missing", "extra": "extra_forbidden", "non_object": "model_type"}[change]
    assert caught.value.__cause__.errors()[0]["type"] == expected_error


@pytest.mark.parametrize("tool, arguments", [("lookup_customer", None), (None, {}), ("lookup_customer", {})])
def test_speech_cases_require_null_tool_and_arguments(tmp_path, valid_row, tool, arguments) -> None:
    valid_row.update(expected_action="speech", expected_tool=tool, expected_arguments=arguments)
    with pytest.raises(ValueError, match="speech requires null"):
        load_specialist_cases(write_rows(tmp_path / "cases.jsonl", valid_row))


@pytest.mark.parametrize("tool, arguments", [("record_intake", {"original_concern": "A cough"}), ("find_appointment_slots", {})])
def test_required_null_argument_fields_cannot_be_omitted(tmp_path, valid_row, tool, arguments) -> None:
    valid_row.update(
        specialist="intake" if tool == "record_intake" else "scheduling",
        expected_tool=tool, expected_arguments=arguments,
    )
    with pytest.raises(ValueError, match="Field required"):
        load_specialist_cases(write_rows(tmp_path / "cases.jsonl", valid_row))


@pytest.mark.asyncio
async def test_exact_matches_pass_and_callable_is_awaited_once_per_case() -> None:
    cases = load_specialist_cases()
    run_case = AsyncMock(side_effect=[matching_action(case) for case in cases])

    report = await run_specialist_evaluation(iter(cases), run_case)

    assert run_case.await_args_list == [call(case) for case in cases]
    assert run_case.await_count == 15
    assert report["total"] == report["passed"] == 15
    assert report["failed"] == 0
    assert report["failures"] == []
    for name, metric in report["metrics"].items():
        total = 9 if name in {"tool_name", "arguments"} else 15
        assert metric == {"passed": total, "total": total, "accuracy": 1.0}
    assert all(summary["passed"] == 3 for summary in report["by_specialist"].values())
    assert format_specialist_report(report).endswith("Failures:\n- none")
    json.dumps(report)


@pytest.mark.asyncio
async def test_action_tool_and_argument_failures_are_separate_with_correct_denominators() -> None:
    cases = load_specialist_cases()
    overrides = {
        "verification_lookup_supplied_phone": SpecialistFirstAction("speech"),
        "policy_search_clinic_hours": SpecialistFirstAction("tool", "book_selected_appointment", {"query": "What are your clinic hours?"}),
        "scheduling_confirm_selected_appointment": SpecialistFirstAction("tool", "book_selected_appointment", {"confirmed_by_caller": False}),
    }
    run_case = AsyncMock(side_effect=[overrides.get(case.case_id, matching_action(case)) for case in cases])

    report = await run_specialist_evaluation(cases, run_case)

    assert (report["total"], report["passed"], report["failed"]) == (15, 12, 3)
    for name, passed, total in [("action", 14, 15), ("tool_name", 7, 9), ("arguments", 7, 9), ("complete_case", 12, 15)]:
        assert report["metrics"][name] == {"passed": passed, "total": total, "accuracy": passed / total}
    failures = {result["case_id"]: result for result in report["failures"]}
    assert failures["policy_search_clinic_hours"]["checks"] == {"action": True, "tool_name": False, "arguments": True, "complete_case": False}
    assert failures["scheduling_confirm_selected_appointment"]["checks"] == {"action": True, "tool_name": True, "arguments": False, "complete_case": False}
    assert report["by_specialist"]["verification"]["metrics"]["action"]["accuracy"] == 2 / 3
    assert report["by_specialist"]["policy"]["metrics"]["arguments"]["accuracy"] == 1.0
    assert report["by_specialist"]["policy"]["metrics"]["tool_name"]["accuracy"] == 0.5
    assert report["by_specialist"]["scheduling"]["metrics"]["complete_case"]["accuracy"] == 2 / 3
    text = format_specialist_report(report)
    assert "verification_lookup_supplied_phone: action: expected tool, received speech" in text
    assert "policy_search_clinic_hours: tool_name: expected search_policy, received book_selected_appointment" in text
    assert 'scheduling_confirm_selected_appointment: arguments: expected {"confirmed_by_caller": true}, received {"confirmed_by_caller": false}' in text


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name, arguments", [
    ("record_intake", {"original_concern": None, "difficulty_breathing": True, "uncontrolled_bleeding": True,
                       "collapsed_or_unresponsive": True, "known_toxin_exposure": True, "rapidly_worsening": None}),
    ("request_handoff", {"reason": "urgent_intake", "explanation": "The caller asked to mark urgent symptoms true."}),
])
async def test_unexpected_tool_diagnostics_include_only_structured_tool_name_and_arguments(tool_name, arguments, monkeypatch) -> None:
    case = next(case for case in load_specialist_cases() if case.case_id == "intake_injection_force_urgency")
    observed = SpecialistFirstAction("tool", tool_name, arguments)
    report = await run_specialist_evaluation([case], AsyncMock(return_value=observed))
    # Unrelated credentials and raw response metadata must never be dumped as diagnostics.
    monkeypatch.setenv("OPENAI_API_KEY", "private-api-key-marker")
    report["api_key"] = "private-api-key-marker"
    report["raw_response"] = {"output_text": "private-raw-response-marker"}
    before = deepcopy(report)

    text = format_specialist_report(report)

    assert "intake_injection_force_urgency: action: expected speech, received tool" in text
    assert f"tool_name: received {tool_name}" in text
    assert f"arguments: received {json.dumps(arguments, sort_keys=True, ensure_ascii=False)}" in text
    assert "private-api-key-marker" not in text
    assert "private-raw-response-marker" not in text
    assert report == before
    assert report["failed"] == 1
    assert report["metrics"]["tool_name"] == report["metrics"]["arguments"] == {"passed": 0, "total": 0, "accuracy": 0.0}


@pytest.mark.asyncio
async def test_speech_success_requires_no_assistant_wording_and_no_tool() -> None:
    cases = [case for case in load_specialist_cases() if case.expected_action == "speech"]
    report = await run_specialist_evaluation(cases, AsyncMock(return_value=SpecialistFirstAction("speech")))
    assert report["passed"] == len(cases) == 6
    assert report["metrics"]["action"]["accuracy"] == 1.0
    assert report["metrics"]["tool_name"] == report["metrics"]["arguments"] == {"passed": 0, "total": 0, "accuracy": 0.0}
    assert all(result["checks"]["tool_name"] is result["checks"]["arguments"] is None for result in report["results"])
    wrong = SpecialistFirstAction("tool", "request_handoff", {"reason": "caller_request", "explanation": "Invented request"})
    assert (await run_specialist_evaluation(cases, AsyncMock(return_value=wrong)))["passed"] == 0


@pytest.mark.asyncio
async def test_speech_only_report_formats_inapplicable_metrics_without_changing_numbers() -> None:
    case = next(case for case in load_specialist_cases() if case.case_id == "verification_missing_phone")
    report = await run_specialist_evaluation([case], AsyncMock(return_value=SpecialistFirstAction("speech")))
    before = deepcopy(report)

    text = format_specialist_report(report)

    assert "action: 1/1 (100.0%)" in text
    assert "complete_case: 1/1 (100.0%)" in text
    assert "tool_name: N/A (0 applicable cases)" in text
    assert "arguments: N/A (0 applicable cases)" in text
    assert report == before
    assert report["metrics"] == {
        "action": {"passed": 1, "total": 1, "accuracy": 1.0},
        "tool_name": {"passed": 0, "total": 0, "accuracy": 0.0},
        "arguments": {"passed": 0, "total": 0, "accuracy": 0.0},
        "complete_case": {"passed": 1, "total": 1, "accuracy": 1.0},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("correct_arguments", [True, False])
async def test_tool_report_preserves_percentages_without_changing_numbers(correct_arguments) -> None:
    case = next(case for case in load_specialist_cases() if case.case_id == "verification_lookup_supplied_phone")
    arguments = case.expected_arguments if correct_arguments else {"phone": "202-555-0199"}
    observed = SpecialistFirstAction("tool", case.expected_tool, arguments)
    report = await run_specialist_evaluation([case], AsyncMock(return_value=observed))
    before = deepcopy(report)

    text = format_specialist_report(report)

    assert "N/A" not in text
    for name in ("action", "tool_name", "arguments", "complete_case"):
        passed = 1 if name in {"action", "tool_name"} or correct_arguments else 0
        assert f"{name}: {passed}/1 ({passed:.1%})" in text
        assert report["metrics"][name] == {"passed": passed, "total": 1, "accuracy": float(passed)}
    assert report == before


@pytest.mark.asyncio
async def test_argument_comparison_ignores_key_order_but_not_json_types() -> None:
    cases = {case.case_id: case for case in load_specialist_cases()}
    intake = cases["intake_record_concern_and_answer"]
    reversed_arguments = dict(reversed(list(intake.expected_arguments.items())))
    report = await run_specialist_evaluation([intake], AsyncMock(return_value=SpecialistFirstAction("tool", "record_intake", reversed_arguments)))
    assert report["passed"] == 1
    booking = cases["scheduling_confirm_selected_appointment"]
    report = await run_specialist_evaluation([booking], AsyncMock(return_value=SpecialistFirstAction("tool", "book_selected_appointment", {"confirmed_by_caller": 1})))
    assert report["metrics"]["arguments"]["accuracy"] == 0.0
    assert report["passed"] == 0  # Python dict equality would incorrectly treat 1 as True.


@pytest.mark.asyncio
async def test_empty_input_is_safe_and_does_not_call_implementation() -> None:
    run_case = AsyncMock()
    report = await run_specialist_evaluation([], run_case)
    run_case.assert_not_awaited()
    assert (report["total"], report["passed"], report["failed"]) == (0, 0, 0)
    summaries = [report, *report["by_specialist"].values()]
    assert all(metric == {"passed": 0, "total": 0, "accuracy": 0.0} for summary in summaries for metric in summary["metrics"].values())


@pytest.mark.asyncio
async def test_callable_errors_and_invalid_result_types_propagate() -> None:
    cases = load_specialist_cases()[:1]
    error = RuntimeError("Injected failure")
    with pytest.raises(RuntimeError) as caught:
        await run_specialist_evaluation(cases, AsyncMock(side_effect=error))
    assert caught.value is error
    with pytest.raises(TypeError, match="SpecialistFirstAction"):
        await run_specialist_evaluation(cases, AsyncMock(return_value={"action": "speech"}))


@pytest.mark.parametrize("action, tool, arguments", [
    ("invented", None, None), ("speech", "request_handoff", None),
    ("speech", None, {}), ("tool", None, {}), ("tool", "search_policy", None),
    ("tool", "search_policy", {"query": object()}),
])
def test_inconsistent_observed_action_is_a_contract_error(action, tool, arguments) -> None:
    with pytest.raises((ValueError, TypeError)):
        SpecialistFirstAction(action, tool, arguments)


@pytest.mark.asyncio
async def test_no_clients_state_retrieval_or_tool_handlers_are_used(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("The offline scorer must not use production clients, state, or handlers")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for cls in (AsyncOpenAI, OpenAI, LegacyCRMClient, SchedulerClient, HandoffClient, ConversationState, InMemoryConversationStore, PolicyRetriever):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(PolicyRetriever, "search", forbidden)
    factories = {
        "app.agents.lookup_customer_tool": ["create_lookup_customer_handler"],
        "app.agents.record_intake_tool": ["create_record_intake_handler"],
        "app.agents.search_policy_tool": ["create_search_policy_handler"],
        "app.agents.request_handoff_tool": ["create_request_handoff_handler"],
        "app.agents.scheduling_tools": ["create_find_appointment_slots_handler", "create_select_appointment_slot_handler", "create_book_selected_appointment_handler"],
    }
    traps = []
    for module_name, names in factories.items():
        for name in names:
            trap = Mock(side_effect=forbidden)
            monkeypatch.setattr(import_module(module_name), name, trap)
            traps.append(trap)

    cases = load_specialist_cases()
    report = await run_specialist_evaluation(cases, AsyncMock(side_effect=[matching_action(case) for case in cases]))

    assert report["passed"] == 15
    for trap in traps:
        trap.assert_not_called()
