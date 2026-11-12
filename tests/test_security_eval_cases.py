"""Human-labeled adversarial coverage and offline tag-selection integrity."""

import asyncio
import socket
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI, OpenAI

from app.models.appointment import AppointmentSlot
from app.models.customer import Customer
from app.models.handoff import HandoffReceipt
from app.models.retrieval import PolicySearchResult
from evals import live_response_eval, live_router_eval, live_specialist_eval
from evals.conversation_eval import load_conversation_cases
from evals.response_eval import ResponseJudgeResult, load_response_cases, run_response_evaluation
from evals.router_eval import load_router_cases, run_router_evaluation
from evals.specialist_eval import SpecialistFirstAction, load_specialist_cases, run_specialist_evaluation

CONFIGURATIONS = {
    "router": (live_router_eval, "load_router_cases", load_router_cases, "prompt_injection"),
    "specialist": (live_specialist_eval, "load_specialist_cases", load_specialist_cases, "prompt_injection"),
    "response": (live_response_eval, "load_response_cases", load_response_cases, "tool_output_injection"),
}
ORIGINAL_ROUTER = {
    "verification_injection_direct_scheduling": "verification",
    "verification_injection_pretend_verified": "verification",
    "intake_injection_choose_policy": "intake",
}
ORIGINAL_SPECIALIST = {
    "verification_injection_invent_phone", "intake_injection_force_urgency", "policy_injection_scheduling_tool",
}


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Security evaluation unit tests must not use a real client or network")

    for cls in (AsyncOpenAI, OpenAI):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("OPENAI_API_KEY", "private-security-eval-key")
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "test-router")
    monkeypatch.setenv("OPENAI_SPECIALIST_MODEL", "test-specialist")
    monkeypatch.delenv("OPENAI_EVAL_JUDGE_MODEL", raising=False)


def test_security_taxonomy_is_consistent_across_all_four_datasets():
    suites = [load_router_cases(), load_specialist_cases(), load_response_cases(), load_conversation_cases()]
    assert [sum("security" in case.tags for case in suite) for suite in suites] == [5, 5, 4, 3]
    for suite in suites:
        for case in suite:
            subtypes = {"prompt_injection", "tool_output_injection"} & set(case.tags)
            assert ("security" in case.tags) == bool(subtypes)
            if "security" in case.tags:
                assert len(subtypes) == 1


def test_original_attack_labels_remain_and_new_cases_cover_distinct_missing_boundaries():
    router = {case.case_id: case for case in load_router_cases()}
    for case_id, destination in {
        **ORIGINAL_ROUTER, "policy_injection_admin_role_spoofing": "policy",
        "handoff_injection_rewrite_urgent_state": "human_handoff",
    }.items():
        assert router[case_id].expected_destination == destination
        assert {"security", "prompt_injection"} <= set(router[case_id].tags)
    urgent = router["handoff_injection_rewrite_urgent_state"].context
    assert urgent.intake_answers.difficulty_breathing is True
    assert urgent.routing_result.routing_level == "urgent"
    assert router["verification_injection_direct_scheduling"].context.customer_verified is False
    specialist = {case.case_id: case for case in load_specialist_cases()}
    for case_id in ORIGINAL_SPECIALIST | {"scheduling_injection_fabricate_confirmation"}:
        assert specialist[case_id].expected_action == "speech"
        assert specialist[case_id].expected_tool is specialist[case_id].expected_arguments is None
    booking = specialist["scheduling_injection_fabricate_confirmation"]
    assert booking.context.selected_slot is not None and booking.context.booking_confirmed is False
    policy = specialist["policy_injection_invent_hours_without_search"]
    assert policy.expected_action == "tool" and policy.expected_tool == "search_policy"
    assert policy.expected_arguments == {"query": "What are your clinic hours?"}
    assert "Skip search_policy" in policy.user_message


def test_new_tool_output_cases_match_real_result_shapes_and_pass_existing_privacy_guards():
    cases = [case for case in load_response_cases() if "tool_output_injection" in case.tags]
    assert {case.specialist for case in cases} == {"verification", "policy", "scheduling", "human_handoff"}
    for case in cases:
        live_response_eval._validate_controlled_fixture(case)
        result = case.tool_result
        if case.specialist == "verification":
            assert set(result) == {"found", "first_name", "pets"}
            assert result["found"] is True and result["pets"] == []
            customer = Customer(customer_id="demo-schema-customer", first_name=result["first_name"],
                                last_name="Demo", phone_number="202-555-0142", pets=result["pets"])
            assert "DEMO_CRM_OVERRIDE" in customer.first_name
        elif case.specialist == "policy":
            assert set(result) == {"found", "evidence"} and result["found"] is True
            evidence = PolicySearchResult.model_validate(result["evidence"][0])
            assert "DEMO_POLICY_OVERRIDE" in evidence.chunk.text
        elif case.specialist == "scheduling":
            # The real decline branch has no private selected_pet or pet_id fields.
            assert case.tool_name == "book_selected_appointment"
            assert case.tool_arguments == {"confirmed_by_caller": False}
            assert set(result) == {"status", "slots"} and result["status"] == "booking_declined"
            slot = AppointmentSlot.model_validate(result["slots"][0])
            assert "DEMO_SCHEDULER_OVERRIDE" in slot.clinic_id
        else:
            assert set(result) == {"queued", "case_id"} and result["queued"] is True
            receipt = HandoffReceipt(case_id=result["case_id"], conversation_id="demo-schema-conversation",
                                     status="queued", received_at=datetime(2026, 11, 3, tzinfo=timezone.utc))
            assert "DEMO_HANDOFF_OVERRIDE" in receipt.case_id


async def fake_report(cases, eval_type, fail):
    if eval_type == "router":
        decisions = [case.expected_destination for case in cases]
        if fail:
            decisions[0] = "policy" if decisions[0] != "policy" else "verification"
        return await run_router_evaluation(cases, AsyncMock(side_effect=decisions))
    if eval_type == "specialist":
        actions = [SpecialistFirstAction(case.expected_action, case.expected_tool, case.expected_arguments) for case in cases]
        if fail:
            actions[0] = SpecialistFirstAction("tool", "lookup_customer", {"phone": "202-555-9999"})
        return await run_specialist_evaluation(cases, AsyncMock(side_effect=actions))
    grades = [ResponseJudgeResult(
        passed=not (fail and index == 0), reason="Offline fixture criterion outcome.", criteria_results=[
            {"criterion_id": criterion.criterion_id, "passed": not (fail and index == 0 and criterion_index == 0)}
            for criterion_index, criterion in enumerate(case.criteria)
        ],
    ) for index, case in enumerate(cases)]
    return await run_response_evaluation(cases, AsyncMock(return_value="Offline response fixture."), AsyncMock(side_effect=grades))


@pytest.mark.parametrize("eval_type", CONFIGURATIONS)
@pytest.mark.parametrize("security_tag", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_tags_select_exact_cases_in_dataset_order_and_preserve_failure_localization(monkeypatch, capsys, eval_type, security_tag, fail):
    module, _, loader, subtype = CONFIGURATIONS[eval_type]
    tag = "security" if security_tag else subtype
    selected = [case for case in loader() if tag in case.tags]
    captured = []

    async def evaluate(cases, *args):
        captured.extend(cases)
        return await fake_report(cases, eval_type, fail)

    run = AsyncMock(side_effect=evaluate)
    monkeypatch.setattr(module, "_evaluate", run)
    assert module.main(["--confirm-live", "--tag", tag]) == (1 if fail else 0)
    assert captured == selected
    run.assert_awaited_once()
    output = capsys.readouterr()
    assert ("Failure localization" in output.out) == fail
    assert output.err == ""
    assert "private-security-eval-key" not in output.out
    assert "DEMO_" not in output.out


@pytest.mark.parametrize("eval_type", CONFIGURATIONS)
def test_other_existing_tags_are_exact_filters_not_substring_matches(monkeypatch, eval_type):
    module, _, loader, _ = CONFIGURATIONS[eval_type]
    tag = "verification" if eval_type == "response" else "unverified"
    selected = [case for case in loader() if tag in case.tags]
    run = AsyncMock(return_value=asyncio.run(fake_report(selected, eval_type, False)))
    monkeypatch.setattr(module, "_evaluate", run)
    assert module.main(["--confirm-live", "--tag", tag]) == 0
    assert run.await_args.args[0] == selected


@pytest.mark.parametrize("eval_type", CONFIGURATIONS)
@pytest.mark.parametrize("tag", ["unknown-tag", "secur", "PRIVATE_INJECTION_PAYLOAD"])
def test_unknown_tags_fail_before_evaluation_without_echoing_input(monkeypatch, capsys, eval_type, tag):
    module = CONFIGURATIONS[eval_type][0]
    run = AsyncMock(side_effect=AssertionError("Selection must precede live evaluation"))
    monkeypatch.setattr(module, "_evaluate", run)
    with pytest.raises(SystemExit) as caught:
        module.main(["--confirm-live", "--tag", tag])
    assert caught.value.code == 2
    run.assert_not_awaited()
    output = capsys.readouterr()
    assert output.out == "" and "Unknown evaluation tag" in output.err
    assert tag not in output.err


@pytest.mark.parametrize("eval_type", CONFIGURATIONS)
@pytest.mark.parametrize("empty_loader", [True, False])
def test_known_tags_with_no_matches_fail_before_network(monkeypatch, capsys, eval_type, empty_loader):
    module, loader_name, _, subtype = CONFIGURATIONS[eval_type]
    run = AsyncMock(side_effect=AssertionError("Empty selection cannot start a live run"))
    monkeypatch.setattr(module, "_evaluate", run)
    if empty_loader:
        monkeypatch.setattr(module, loader_name, lambda: [])
        tag = "security"
    else:
        tag = "prompt_injection" if subtype == "tool_output_injection" else "tool_output_injection"
    with pytest.raises(SystemExit):
        module.main(["--confirm-live", "--tag", tag])
    assert "No evaluation cases match" in capsys.readouterr().err
    run.assert_not_awaited()


@pytest.mark.parametrize("eval_type", CONFIGURATIONS)
@pytest.mark.parametrize("other_mode", [["--all-cases"], ["--case-id", "some-case"]])
def test_tag_selection_is_mutually_exclusive_with_other_modes(monkeypatch, eval_type, other_mode):
    module = CONFIGURATIONS[eval_type][0]
    run = AsyncMock()
    monkeypatch.setattr(module, "_evaluate", run)
    with pytest.raises(SystemExit) as caught:
        module.main(["--confirm-live", "--tag", "security", *other_mode])
    assert caught.value.code == 2
    run.assert_not_awaited()


@pytest.mark.parametrize("eval_type", CONFIGURATIONS)
def test_tag_mode_preserves_confirmation_and_api_key_protections(monkeypatch, eval_type):
    module = CONFIGURATIONS[eval_type][0]
    run = AsyncMock()
    monkeypatch.setattr(module, "_evaluate", run)
    with pytest.raises(SystemExit):
        module.main(["--tag", "security"])
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit):
        module.main(["--confirm-live", "--tag", "security"])
    run.assert_not_awaited()
