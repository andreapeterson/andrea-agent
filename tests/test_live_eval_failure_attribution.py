"""All live CLIs append safe localization to existing offline-generated reports."""

import asyncio
import socket
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
from openai import AsyncOpenAI, OpenAI

from evals import live_conversation_eval, live_response_eval, live_router_eval, live_specialist_eval
from evals.conversation_eval import ConversationRunObservation, load_conversation_cases, run_conversation_evaluation
from evals.failure_attribution import attribute_report_failures, format_failure_attribution_report
from evals.response_eval import ResponseJudgeResult, load_response_cases, run_response_evaluation
from evals.router_eval import load_router_cases, run_router_evaluation
from evals.specialist_eval import SpecialistFirstAction, load_specialist_cases, run_specialist_evaluation

MODULES = {
    "router": (live_router_eval, "format_router_report", "router_selection"),
    "specialist": (live_specialist_eval, "format_specialist_report", "tool_arguments"),
    "response": (live_response_eval, "format_response_report", "response_generation"),
    "conversation": (live_conversation_eval, "format_conversation_report", "conversation_state"),
}
FAKE_KEY = "private-live-eval-key"


@pytest.fixture(autouse=True)
def offline_configuration(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Report localization must not construct clients or make network calls")

    for cls in (AsyncOpenAI, OpenAI):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "test-router-model")
    monkeypatch.setenv("OPENAI_SPECIALIST_MODEL", "test-specialist-model")
    monkeypatch.delenv("OPENAI_EVAL_JUDGE_MODEL", raising=False)


async def make_report(eval_type, *, passed, multiple=False):
    if eval_type == "router":
        case = next(case for case in load_router_cases() if case.expected_destination == "policy")
        return await run_router_evaluation([case], AsyncMock(return_value="policy" if passed else "verification"))
    if eval_type == "specialist":
        case = next(case for case in load_specialist_cases() if case.case_id == "verification_lookup_supplied_phone")
        observed = SpecialistFirstAction(action="tool", tool_name="lookup_customer", arguments=case.expected_arguments if passed else {"phone": "202-555-9999"})
        if multiple:
            observed = SpecialistFirstAction(action="speech")
        return await run_specialist_evaluation([case], AsyncMock(return_value=observed))
    if eval_type == "response":
        case = load_response_cases()[0]
        grade = ResponseJudgeResult(
            passed=passed, reason="Fixture criteria checked.", criteria_results=[
                {"criterion_id": criterion.criterion_id, "passed": passed or index > 0}
                for index, criterion in enumerate(case.criteria)
            ],
        )
        return await run_response_evaluation([case], AsyncMock(return_value="Private generated response."), AsyncMock(return_value=grade))
    case = load_conversation_cases()[0]
    observation = ConversationRunObservation(
        specialists_by_turn=case.expected_specialists_by_turn, tools_by_turn=case.expected_tools_by_turn,
        final_phase=case.expected_final_phase if passed else "completed", verified_customer=False,
        booking_confirmed=False, handoff_queued=False,
    )
    return await run_conversation_evaluation([case], AsyncMock(return_value=observation))


@pytest.mark.parametrize("eval_type", MODULES)
@pytest.mark.parametrize("passed", [True, False])
def test_original_report_output_and_exit_code_are_preserved_with_explicit_localization_type(monkeypatch, capsys, eval_type, passed):
    module, formatter_name, component = MODULES[eval_type]
    source = asyncio.run(make_report(eval_type, passed=passed))
    before = deepcopy(source)
    original_text = getattr(module, formatter_name)(source)
    evaluate = AsyncMock(return_value=source)
    attribution = Mock(wraps=attribute_report_failures)
    localization_formatter = Mock(wraps=format_failure_attribution_report)
    monkeypatch.setattr(module, "_evaluate", evaluate)
    monkeypatch.setattr(module, "attribute_report_failures", attribution)
    monkeypatch.setattr(module, "format_failure_attribution_report", localization_formatter)

    status = module.main(["--confirm-live", "--case-id", source["results"][0]["case_id"]])

    assert status == (0 if passed else 1)
    evaluate.assert_awaited_once()
    attribution.assert_called_once_with(source, eval_type=eval_type)
    assert source == before
    output = capsys.readouterr()
    assert output.err == ""
    if passed:
        assert output.out == original_text + "\n"
        localization_formatter.assert_not_called()
        assert "Failure localization" not in output.out
    else:
        items = attribute_report_failures(source, eval_type=eval_type)
        localization_formatter.assert_called_once_with(items)
        assert output.out == original_text + "\n\n" + format_failure_attribution_report(items) + "\n"
        assert f"{component}: 1 (cases: {source['results'][0]['case_id']})" in output.out
        localization_text = output.out.split("\n\nFailure localization\n", 1)[1]
        assert "Private generated response." not in localization_text


def test_multiple_components_follow_existing_deterministic_formatter_order(monkeypatch, capsys):
    source = asyncio.run(make_report("specialist", passed=False, multiple=True))
    source["results"][0]["checks"] = dict(reversed(list(source["results"][0]["checks"].items())))
    monkeypatch.setattr(live_specialist_eval, "_evaluate", AsyncMock(return_value=source))
    assert live_specialist_eval.main(["--confirm-live", "--case-id", source["results"][0]["case_id"]]) == 1
    localization = capsys.readouterr().out.split("\n\nFailure localization\n", 1)[1]
    assert localization.splitlines()[0] == "Total attributions: 3"
    assert [line.split(":", 1)[0] for line in localization.splitlines()[1:]] == [
        "specialist_action", "tool_selection", "tool_arguments",
    ]


@pytest.mark.parametrize("eval_type", MODULES)
def test_inconsistent_reports_abort_before_any_report_or_sensitive_fields_are_printed(monkeypatch, capsys, eval_type):
    module, _, _ = MODULES[eval_type]
    source = asyncio.run(make_report(eval_type, passed=False))
    source["total"] += 1  # Real attribution validation must reject this report.
    source["api_key"] = FAKE_KEY
    source["results"][0].update(
        response="PRIVATE_GENERATED_RESPONSE", reason="PRIVATE_JUDGE_PAYLOAD",
        actual_arguments={"phone": "PRIVATE_PHONE"}, tool_result={"customer_id": "PRIVATE_CUSTOMER"},
        failure_details=["PRIVATE_EXCEPTION", FAKE_KEY],
    )
    evaluate = AsyncMock(return_value=source)
    original_formatter = Mock(side_effect=AssertionError("Invalid reports must not be formatted"))
    monkeypatch.setattr(module, "_evaluate", evaluate)
    monkeypatch.setattr(module, MODULES[eval_type][1], original_formatter)

    assert module.main(["--confirm-live", "--case-id", source["results"][0]["case_id"]]) == 1

    original_formatter.assert_not_called()
    evaluate.assert_awaited_once()
    output = capsys.readouterr()
    assert output.out == ""
    assert "failure localization failed; no report was produced." in output.err
    assert FAKE_KEY not in output.err and "PRIVATE" not in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize("eval_type", MODULES)
@pytest.mark.parametrize("stage", ["attribution", "formatter"])
def test_localization_errors_never_echo_upstream_exceptions_or_partial_reports(monkeypatch, capsys, eval_type, stage):
    module, _, _ = MODULES[eval_type]
    source = asyncio.run(make_report(eval_type, passed=False))
    monkeypatch.setattr(module, "_evaluate", AsyncMock(return_value=source))
    failed_function = "attribute_report_failures" if stage == "attribution" else "format_failure_attribution_report"
    monkeypatch.setattr(module, failed_function, Mock(side_effect=ValueError(
        FAKE_KEY + " PRIVATE_PROMPT PRIVATE_RESPONSE PRIVATE_TOOL_ARGUMENTS PRIVATE_RESULT UPSTREAM_EXCEPTION",
    )))
    assert module.main(["--confirm-live", "--case-id", source["results"][0]["case_id"]]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "failure localization failed; no report was produced." in output.err
    assert FAKE_KEY not in output.err and "PRIVATE" not in output.err and "UPSTREAM" not in output.err
