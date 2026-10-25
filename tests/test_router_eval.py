"""Verify the evaluation harness, not real model routing accuracy."""

import json
import sys
from collections import Counter
from types import ModuleType
from unittest.mock import AsyncMock, call

import pytest
from openai import AsyncOpenAI

from app.agents.prompt_context import PromptContext
from app.integrations import HandoffClient, LegacyCRMClient, SchedulerClient
from app.models.conversation import ConversationState
from app.services.conversation_store import InMemoryConversationStore
from evals.router_eval import (
    DEFAULT_CASES_PATH,
    DESTINATIONS,
    format_router_report,
    load_router_cases,
    main,
    run_router_evaluation,
)


@pytest.fixture
def valid_row():
    return json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8").splitlines()[0])


def write_rows(path, *rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_dataset_loads_unique_cases_with_five_per_destination() -> None:
    cases = load_router_cases()

    assert len(cases) == 25
    assert len({case.case_id for case in cases}) == 25
    assert Counter(case.expected_destination for case in cases) == {destination: 5 for destination in DESTINATIONS}
    for case in cases:
        assert isinstance(case.context, PromptContext)
        assert case.user_message and case.reason and case.tags
        assert all(message.content != case.user_message for message in case.context.recent_messages)


def test_dataset_has_injection_urgent_unverified_scheduling_and_ambiguous_cases() -> None:
    cases = load_router_cases()
    injections = [case for case in cases if "prompt_injection" in case.tags]
    assert len(injections) >= 3
    assert {case.expected_destination for case in injections} == {"verification", "intake"}
    assert any(
        case.context.routing_result is not None
        and case.context.routing_result.routing_level == "urgent"
        and case.expected_destination == "human_handoff"
        for case in cases
    )
    assert any(
        "unverified_scheduling" in case.tags
        and case.context.customer_verified is False
        and case.expected_destination == "verification"
        for case in cases
    )
    assert any("ambiguous" in case.tags for case in cases)


def test_whitespace_lines_are_skipped_and_metadata_is_trimmed(tmp_path, valid_row) -> None:
    valid_row["case_id"] = "  trimmed_id  "
    valid_row["user_message"] = "  Can I schedule?  "
    valid_row["reason"] = "  Verification comes first.  "
    valid_row["tags"] = ["  unverified  "]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n \t\n" + json.dumps(valid_row) + "\n\n", encoding="utf-8")

    cases = load_router_cases(path)

    assert len(cases) == 1
    assert cases[0].case_id == "trimmed_id"
    assert cases[0].user_message == "Can I schedule?"
    assert cases[0].reason == "Verification comes first."
    assert cases[0].tags == ["unverified"]


@pytest.mark.parametrize("content", ["{broken", "not JSON", '{"context": NaN}', '{"context": Infinity}'])
def test_invalid_json_reports_file_and_line(tmp_path, content) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("\n" + content, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_router_cases(path)
    assert str(caught.value).startswith(f"{path}:2:")


@pytest.mark.parametrize("context", [{"phone_number": "fictional"}, {"customer_verified": "not-a-bool"}, {"routing_result": {"routing_level": "invented", "next_action": "create_handoff"}}, None])
def test_context_is_validated_by_existing_prompt_context(tmp_path, valid_row, context) -> None:
    valid_row["context"] = context
    with pytest.raises(ValueError, match="PromptContext"):
        load_router_cases(write_rows(tmp_path / "cases.jsonl", valid_row))


def test_duplicate_normalized_case_ids_are_rejected(tmp_path, valid_row) -> None:
    duplicate = {**valid_row, "case_id": " " + valid_row["case_id"] + " "}
    with pytest.raises(ValueError, match="duplicate case_id"):
        load_router_cases(write_rows(tmp_path / "cases.jsonl", valid_row, duplicate))


@pytest.mark.parametrize("destination", ["booking", "Policy", "", None, ["policy"]])
def test_invalid_expected_destinations_are_rejected(tmp_path, valid_row, destination) -> None:
    valid_row["expected_destination"] = destination
    with pytest.raises(ValueError, match="invalid expected_destination"):
        load_router_cases(write_rows(tmp_path / "cases.jsonl", valid_row))


@pytest.mark.parametrize("field, value", [
    ("case_id", ""), ("case_id", " \t "), ("case_id", None), ("case_id", 7),
    ("user_message", ""), ("user_message", " \n "), ("user_message", None),
    ("reason", ""), ("reason", "  "), ("reason", None),
    ("tags", []), ("tags", [""]), ("tags", ["  "]),
    ("tags", ["valid", ""]), ("tags", [None]), ("tags", [7]), ("tags", "routine"),
])
def test_blank_or_invalid_metadata_is_rejected(tmp_path, valid_row, field, value) -> None:
    valid_row[field] = value
    with pytest.raises(ValueError, match="nonblank|nonempty"):
        load_router_cases(write_rows(tmp_path / "cases.jsonl", valid_row))


@pytest.mark.parametrize("change", ["missing", "extra", "non_object"])
def test_invalid_case_shape_is_rejected(tmp_path, valid_row, change) -> None:
    if change == "missing":
        del valid_row["reason"]
    elif change == "extra":
        valid_row["unexpected"] = "value"
    else:
        valid_row = [valid_row]
    with pytest.raises(ValueError, match="exactly these fields"):
        load_router_cases(write_rows(tmp_path / "cases.jsonl", valid_row))


@pytest.mark.asyncio
async def test_exact_matches_pass_and_route_callable_receives_only_context_and_message_once_per_case() -> None:
    cases = load_router_cases()
    route_case = AsyncMock(side_effect=[case.expected_destination for case in cases])

    report = await run_router_evaluation(iter(cases), route_case)

    assert route_case.await_count == 25
    assert route_case.await_args_list == [call(case.context, case.user_message) for case in cases]
    assert report["total"] == report["passed"] == 25
    assert report["failed"] == 0
    assert report["accuracy"] == 1.0
    assert report["failures"] == []
    assert report["results"][0] == {
        "case_id": cases[0].case_id,
        "expected_destination": cases[0].expected_destination,
        "actual_destination": cases[0].expected_destination,
        "passed": True,
        "tags": cases[0].tags,
    }
    assert format_router_report(report).endswith("Failures:\n- none")


@pytest.mark.asyncio
async def test_mismatches_score_overall_and_by_expected_destination_and_format_readable_failures() -> None:
    cases = load_router_cases()
    mistakes = {"intake_followup_no": "verification", "scheduling_select_second_offered_slot": "policy"}
    route_case = AsyncMock(side_effect=[mistakes.get(case.case_id, case.expected_destination) for case in cases])

    report = await run_router_evaluation(cases, route_case)

    assert report["total"] == 25
    assert report["passed"] == 23
    assert report["failed"] == 2
    assert report["accuracy"] == pytest.approx(0.92)
    for destination, group in report["by_destination"].items():
        expected_passed = 4 if destination in {"intake", "scheduling"} else 5
        assert group == {"total": 5, "passed": expected_passed, "failed": 5 - expected_passed, "accuracy": expected_passed / 5}
    text = format_router_report(report)
    assert "Total: 25\nPassed: 23\nFailed: 2\nAccuracy: 92.0%" in text
    assert "intake: 4/5" in text
    assert "scheduling: 4/5" in text
    assert "- intake_followup_no: expected intake, received verification" in text
    assert "- scheduling_select_second_offered_slot: expected scheduling, received policy" in text
    assert {failure["case_id"] for failure in report["failures"]} == set(mistakes)
    json.dumps(report)  # Reports contain plain JSON-safe values.


@pytest.mark.asyncio
@pytest.mark.parametrize("actual", ["Verification", " verification ", "unexpected_destination"])
async def test_grading_uses_exact_strings_without_normalizing_actual_predictions(actual) -> None:
    report = await run_router_evaluation(load_router_cases()[:1], AsyncMock(return_value=actual))
    assert report["passed"] == 0
    assert report["failures"][0]["actual_destination"] == actual


@pytest.mark.asyncio
async def test_empty_evaluation_does_not_call_router_or_divide_by_zero() -> None:
    route_case = AsyncMock()
    report = await run_router_evaluation([], route_case)
    route_case.assert_not_awaited()
    assert report["total"] == report["passed"] == report["failed"] == 0
    assert report["accuracy"] == 0.0
    assert all(group["accuracy"] == 0.0 for group in report["by_destination"].values())


@pytest.mark.asyncio
async def test_routing_errors_propagate_instead_of_becoming_successful_scores() -> None:
    error = RuntimeError("Injected routing failure")
    with pytest.raises(RuntimeError) as caught:
        await run_router_evaluation(load_router_cases(), AsyncMock(side_effect=error))
    assert caught.value is error


@pytest.mark.asyncio
async def test_non_string_route_result_is_a_callable_contract_error() -> None:
    with pytest.raises(TypeError, match="destination string"):
        await run_router_evaluation(load_router_cases()[:1], AsyncMock(return_value=None))


@pytest.mark.asyncio
async def test_evaluation_requires_no_production_state_clients_or_external_services(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("The offline evaluation must not construct production clients or state")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for cls in (AsyncOpenAI, LegacyCRMClient, SchedulerClient, HandoffClient, ConversationState, InMemoryConversationStore):
        monkeypatch.setattr(cls, "__init__", forbidden)
    cases = load_router_cases()
    report = await run_router_evaluation(cases, AsyncMock(return_value="policy"))
    assert report["total"] == 25
    assert report["passed"] == 5


def test_command_line_uses_an_explicitly_injected_fake_callable(monkeypatch, capsys) -> None:
    module = ModuleType("fake_router_for_eval")
    route_case = AsyncMock(return_value="policy")
    module.route_case = route_case
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["router_eval", "--route-case", "fake_router_for_eval:route_case"])

    main()

    assert route_case.await_count == 25
    text = capsys.readouterr().out
    assert "Router evaluation\nTotal: 25\nPassed: 5\nFailed: 20\nAccuracy: 20.0%" in text


def test_command_line_requires_module_function_syntax(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["router_eval", "--route-case", "missing_separator"])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    assert "module:function" in capsys.readouterr().err
