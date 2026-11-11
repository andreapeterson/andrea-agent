"""Localize deterministic mismatches without copying content-bearing report fields."""

import json
import socket
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from evals.failure_attribution import (
    CONVERSATION_CHECKS,
    REVIEW_TARGETS,
    FailureAttribution,
    attribute_conversation_result,
    attribute_report_failures,
    attribute_response_result,
    attribute_router_result,
    attribute_specialist_result,
    format_failure_attribution_report,
    summarize_failure_attributions,
)


@pytest.fixture(autouse=True)
def forbid_network_and_openai_clients(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Failure attribution must stay pure and offline")

    for cls in (OpenAI, AsyncOpenAI):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def router_row(*, passed=False, case_id="demo-router"):
    return {"case_id": case_id, "passed": passed, "expected_destination": "policy",
            "actual_destination": "policy" if passed else "verification"}


def specialist_row(*, failed=("arguments",), case_id="demo-specialist"):
    checks = {name: name not in failed for name in ("action", "tool_name", "arguments")}
    checks["complete_case"] = all(checks.values())
    return {"case_id": case_id, "passed": checks["complete_case"], "checks": checks}


def response_row(*, failed=("grounded",), case_id="demo-response"):
    criteria = [{"criterion_id": name, "description": f"Safe human criterion: {name}.", "passed": name not in failed}
                for name in ("grounded", "concise")]
    passed = all(criterion["passed"] for criterion in criteria)
    return {"case_id": case_id, "passed": passed, "judge_passed": passed, "criteria_results": criteria}


def conversation_row(*, failed=("final_phase",), case_id="demo-conversation"):
    checks = {name: name not in failed for name, _ in CONVERSATION_CHECKS}
    checks["complete_case"] = all(checks.values())
    return {"case_id": case_id, "passed": checks["complete_case"], "checks": checks}


def report(*rows):
    passed = sum(row["passed"] for row in rows)
    return {"total": len(rows), "passed": passed, "failed": len(rows) - passed, "results": list(rows)}


@pytest.mark.parametrize("eval_type, row", [
    ("router", router_row(passed=True)), ("specialist", specialist_row(failed=())),
    ("response", response_row(failed=())), ("conversation", conversation_row(failed=())),
])
def test_passing_and_empty_reports_produce_an_immutable_empty_sequence(eval_type, row):
    assert attribute_report_failures(report(row), eval_type=eval_type) == ()
    assert attribute_report_failures(report(), eval_type=eval_type) == ()


def test_router_destination_mismatch_is_localized_without_copying_destination_text():
    row = router_row()
    row["actual_destination"] = "PRIVATE_GENERATED_DESTINATION"
    attribution, = attribute_router_result(row)
    assert attribution.component == "router_selection"
    assert attribution.eval_type == "router" and attribution.case_id == row["case_id"]
    assert "PRIVATE_GENERATED_DESTINATION" not in attribution.model_dump_json()
    assert attribution.suggested_review_target == REVIEW_TARGETS["router_selection"]
    row["actual_destination"] = ""
    assert attribute_router_result(row)[0].component == "router_selection"


@pytest.mark.parametrize("check, component", [
    ("action", "specialist_action"), ("tool_name", "tool_selection"), ("arguments", "tool_arguments"),
])
def test_each_specialist_check_has_its_own_component(check, component):
    attribution, = attribute_specialist_result(specialist_row(failed=(check,)))
    assert attribution.component == component
    assert check in attribution.observed_failure
    assert attribution.suggested_review_target == REVIEW_TARGETS[component]


def test_one_specialist_case_produces_multiple_attributions_in_fixed_check_order():
    items = attribute_specialist_result(specialist_row(failed=("arguments", "tool_name", "action")))
    assert isinstance(items, tuple)
    assert [item.component for item in items] == ["specialist_action", "tool_selection", "tool_arguments"]
    assert len({item.case_id for item in items}) == 1
    with pytest.raises(ValidationError, match="frozen"):
        items[0].component = "router_selection"


def test_speech_cases_leave_nonapplicable_tool_checks_unattributed():
    row = specialist_row(failed=("action",))
    row["checks"].update(tool_name=None, arguments=None)
    items = attribute_specialist_result(row)
    assert [item.component for item in items] == ["specialist_action"]


@pytest.mark.parametrize("check, component", CONVERSATION_CHECKS)
def test_each_conversation_check_maps_to_the_observed_layer(check, component):
    attribution, = attribute_conversation_result(conversation_row(failed=(check,)))
    assert attribution.component == component
    assert check in attribution.observed_failure
    assert "expected" not in attribution.observed_failure


def test_multiple_conversation_failures_remain_distinct_even_in_the_same_component():
    items = attribute_conversation_result(conversation_row(failed=("tool_sequence", "specialist_sequence", "final_phase", "booking_confirmed")))
    assert [item.component for item in items] == ["orchestration", "orchestration", "conversation_state", "conversation_state"]
    assert len({item.observed_failure for item in items}) == 4


def test_response_criteria_use_only_ids_and_safe_human_descriptions():
    row = response_row(failed=("grounded", "concise"))
    row.update(response="PRIVATE_MODEL_TEXT", reason="PRIVATE_JUDGE_PAYLOAD")
    items = attribute_response_result(row)
    assert [item.observed_failure for item in items] == [
        "Failure observed in response criterion concise: Safe human criterion: concise.",
        "Failure observed in response criterion grounded: Safe human criterion: grounded.",
    ]
    assert all(item.component == "response_generation" for item in items)
    assert "PRIVATE" not in json.dumps([item.model_dump() for item in items])


def test_failed_overall_response_grade_without_failed_criteria_has_safe_localization():
    row = response_row(failed=())
    row.update(passed=False, judge_passed=False, reason="PRIVATE_JUDGE_REASON")
    item, = attribute_response_result(row)
    assert item.component == "response_generation"
    assert "no individual criterion was marked failed" in item.observed_failure
    assert "PRIVATE" not in item.model_dump_json()


def test_order_is_independent_of_result_check_and_criterion_insertion_order():
    rows = [specialist_row(case_id="z-case", failed=("tool_name", "action")), specialist_row(case_id="a-case")]
    expected = attribute_report_failures(report(*rows), eval_type="specialist")
    rows.reverse()
    for row in rows:
        row["checks"] = dict(reversed(list(row["checks"].items())))
    assert attribute_report_failures(report(*rows), eval_type="specialist") == expected
    assert [item.case_id for item in expected] == ["a-case", "z-case", "z-case"]
    row = response_row(failed=("grounded", "concise"))
    expected = attribute_report_failures(report(row), eval_type="response")
    row["criteria_results"].reverse()
    assert attribute_report_failures(report(row), eval_type="response") == expected


@pytest.mark.parametrize("eval_type, row", [
    ("router", router_row()), ("specialist", specialist_row()),
    ("response", response_row()), ("conversation", conversation_row()),
])
def test_reports_are_not_mutated_and_sensitive_fields_are_not_copied(eval_type, row):
    row.update(
        response="PRIVATE_GENERATED_RESPONSE", reason="PRIVATE_JUDGE_REASON",
        expected_arguments={"phone": "PRIVATE_PHONE"}, actual_arguments={"query": "PRIVATE_QUERY"},
        tool_result={"customer_id": "PRIVATE_CUSTOMER"}, caller_message="PRIVATE_CALLER",
        failure_details=["PRIVATE_EXCEPTION_AND_ARGUMENTS"],
    )
    source = report(row)
    source.update(api_key="PRIVATE_CREDENTIAL", raw_response={"text": "PRIVATE_RAW"}, failures=[row])
    before = deepcopy(source)
    items = attribute_report_failures(source, eval_type=eval_type)
    assert source == before
    assert "PRIVATE" not in json.dumps([item.model_dump() for item in items])
    output = format_failure_attribution_report(items)
    assert "PRIVATE" not in output
    assert "Safe human criterion" not in output
    assert "Review " not in output  # Counts and case IDs only.


def test_payload_objects_need_not_be_serializable_or_inspected():
    class PrivatePayload:
        def __repr__(self):
            raise AssertionError("Private payloads must not be inspected")

    row = specialist_row()
    row.update(actual_arguments=PrivatePayload(), expected_arguments=PrivatePayload(), failure_details=PrivatePayload())
    assert attribute_report_failures(report(row), eval_type="specialist")[0].component == "tool_arguments"


@pytest.mark.parametrize("kind", ["unknown", "", None, [], 7])
def test_report_type_is_explicit_and_unknown_types_are_rejected(kind):
    with pytest.raises(ValueError, match="Unsupported eval_type"):
        attribute_report_failures(report(), eval_type=kind)


def test_no_report_type_is_automatically_inferred():
    with pytest.raises(TypeError, match="eval_type"):
        attribute_report_failures(report(router_row()))


@pytest.mark.parametrize("change", [
    "missing_results", "results_not_list", "row_not_mapping", "missing_case_id", "blank_case_id",
    "missing_passed", "nonboolean_passed", "duplicate_case_id", "missing_total", "negative_total",
    "boolean_total", "incorrect_count", "missing_destination", "invalid_actual_destination",
    "inconsistent_passed",
])
def test_malformed_router_reports_fail_clearly_without_echoing_payloads(change):
    row = router_row()
    source = report(row)
    if change == "missing_results":
        del source["results"]
    elif change == "results_not_list":
        source["results"] = {}
    elif change == "row_not_mapping":
        source["results"] = [None]
    elif change == "missing_case_id":
        del row["case_id"]
    elif change == "blank_case_id":
        row["case_id"] = " "
    elif change == "missing_passed":
        del row["passed"]
    elif change == "nonboolean_passed":
        row["passed"] = "PRIVATE_FALSE"
    elif change == "duplicate_case_id":
        source = report(row, {**row, "case_id": " " + row["case_id"] + " "})
    elif change == "missing_total":
        del source["total"]
    elif change == "negative_total":
        source["total"] = -1
    elif change == "boolean_total":
        source["total"] = True
    elif change == "incorrect_count":
        source["failed"] = 0
    elif change == "missing_destination":
        del row["expected_destination"]
    elif change == "invalid_actual_destination":
        row["actual_destination"] = {"private": "PRIVATE_PAYLOAD"}
    else:
        row["actual_destination"] = "policy"
    with pytest.raises(ValueError) as caught:
        attribute_report_failures(source, eval_type="router")
    assert str(caught.value)
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize("eval_type, row", [("specialist", specialist_row()), ("conversation", conversation_row())])
@pytest.mark.parametrize("change", ["missing_checks", "missing_check", "invalid_flag", "inconsistent_complete", "inconsistent_passed"])
def test_invalid_check_reports_are_rejected(eval_type, row, change):
    row = deepcopy(row)
    first = "action" if eval_type == "specialist" else "specialist_sequence"
    if change == "missing_checks":
        del row["checks"]
    elif change == "missing_check":
        del row["checks"][first]
    elif change == "invalid_flag":
        row["checks"][first] = "PRIVATE_TRUE"
    elif change == "inconsistent_complete":
        row["checks"]["complete_case"] = True
    else:
        row["passed"] = True
    with pytest.raises(ValueError):
        attribute_report_failures(report(row), eval_type=eval_type)


@pytest.mark.parametrize("change", ["missing_criteria", "empty_criteria", "missing_description", "blank_id", "duplicate_id", "invalid_flag", "missing_grade", "inconsistent_grade"])
def test_invalid_response_criterion_reports_are_rejected(change):
    row = response_row()
    if change == "missing_criteria":
        del row["criteria_results"]
    elif change == "empty_criteria":
        row["criteria_results"] = []
    elif change == "missing_description":
        del row["criteria_results"][0]["description"]
    elif change == "blank_id":
        row["criteria_results"][0]["criterion_id"] = " "
    elif change == "duplicate_id":
        row["criteria_results"].append(row["criteria_results"][0])
    elif change == "invalid_flag":
        row["criteria_results"][0]["passed"] = 0
    elif change == "missing_grade":
        del row["judge_passed"]
    else:
        row.update(passed=True, judge_passed=True)
    with pytest.raises(ValueError):
        attribute_report_failures(report(row), eval_type="response")


def test_aggregate_counts_and_formatter_are_stable_and_count_observations_not_unique_cases():
    items = (
        *attribute_specialist_result(specialist_row(case_id="z-case")),
        *attribute_specialist_result(specialist_row(case_id="a-case")),
        *attribute_response_result(response_row()),
        *attribute_conversation_result(conversation_row()),
    )
    assert summarize_failure_attributions(items) == {
        "total_attributions": 4,
        "by_component": {"tool_arguments": 2, "response_generation": 1, "conversation_state": 1},
    }
    output = format_failure_attribution_report(iter(reversed(items)))
    assert output == format_failure_attribution_report(items)
    assert "Total attributions: 4" in output
    assert "tool_arguments: 2 (cases: a-case, z-case)" in output
    assert "response_generation: 1 (cases: demo-response)" in output
    assert summarize_failure_attributions(()) == {"total_attributions": 0, "by_component": {}}
    assert format_failure_attribution_report(()) == "Failure localization\nTotal attributions: 0\n- none"


@pytest.mark.parametrize("function", [summarize_failure_attributions, format_failure_attribution_report])
def test_aggregate_and_formatter_require_typed_attributions(function):
    with pytest.raises(TypeError, match="FailureAttribution"):
        function([{}])


def test_model_rejects_private_extra_fields_blank_metadata_and_unknown_literals():
    payload = attribute_router_result(router_row())[0].model_dump()
    for change in ({"case_id": " "}, {"eval_type": "other"}, {"component": "prompt_root_cause"},
                   {"observed_failure": " "}, {"suggested_review_target": " "}, {"raw_response": "PRIVATE"}):
        with pytest.raises(ValidationError):
            FailureAttribution.model_validate({**payload, **change})


@pytest.mark.asyncio
async def test_existing_four_harness_report_formats_are_supported_without_modification():
    from evals.router_eval import load_router_cases, run_router_evaluation
    from evals.specialist_eval import SpecialistFirstAction, load_specialist_cases, run_specialist_evaluation
    from evals.response_eval import ResponseJudgeResult, load_response_cases, run_response_evaluation
    from evals.conversation_eval import ConversationRunObservation, load_conversation_cases, run_conversation_evaluation

    router_case = load_router_cases()[0]
    wrong_destination = "verification" if router_case.expected_destination != "verification" else "policy"
    router_report = await run_router_evaluation([router_case], AsyncMock(return_value=wrong_destination))
    specialist_case = next(case for case in load_specialist_cases() if case.expected_action == "tool")
    specialist_report = await run_specialist_evaluation([specialist_case], AsyncMock(return_value=SpecialistFirstAction(action="speech")))
    response_case = load_response_cases()[0]
    grade = ResponseJudgeResult(passed=False, reason="PRIVATE_JUDGE_REASON", criteria_results=[
        {"criterion_id": criterion.criterion_id, "passed": index != 0} for index, criterion in enumerate(response_case.criteria)
    ])
    response_report = await run_response_evaluation([response_case], AsyncMock(return_value="PRIVATE_RESPONSE"), AsyncMock(return_value=grade))
    conversation_report = await run_conversation_evaluation(load_conversation_cases()[:1], AsyncMock(return_value=ConversationRunObservation(
        specialists_by_turn=["verification"], tools_by_turn=[[]], final_phase="verifying_customer",
        verified_customer=False, booking_confirmed=False, handoff_queued=False,
    )))
    for eval_type, source in (
        ("router", router_report), ("specialist", specialist_report),
        ("response", response_report), ("conversation", conversation_report),
    ):
        before = deepcopy(source)
        items = attribute_report_failures(source, eval_type=eval_type)
        assert items and all(item.eval_type == eval_type for item in items)
        assert source == before
        assert "PRIVATE" not in json.dumps([item.model_dump() for item in items])
