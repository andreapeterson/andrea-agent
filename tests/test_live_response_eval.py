"""Exercise real prompt/tool-loop wiring with fake OpenAI clients and fixture judgments."""

import asyncio
import json
import socket
from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from openai import AsyncOpenAI

from app.agents.lookup_customer_tool import LOOKUP_CUSTOMER_TOOL
from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.record_intake_tool import RECORD_INTAKE_TOOL
from app.agents.request_handoff_tool import REQUEST_HANDOFF_TOOL
from app.agents.scheduling_tools import BOOK_SELECTED_APPOINTMENT_TOOL, FIND_APPOINTMENT_SLOTS_TOOL, SELECT_APPOINTMENT_SLOT_TOOL
from app.agents.search_policy_tool import SEARCH_POLICY_TOOL
from app.integrations import HandoffClient, LegacyCRMClient, SchedulerClient
from app.models.conversation import ConversationState
from app.services.conversation_store import InMemoryConversationStore
from app.services.policy_retriever import PolicyRetriever
from evals import live_response_eval
from evals.response_eval import ResponseCriterion, ResponseJudgeResult, load_response_cases, run_response_evaluation

CASES = load_response_cases()
SMOKE_CASE = "verification_customer_not_found"
FAKE_KEY = "fake-api-key-never-print"
PRIVATE = "private-prompt-upstream-marker"
CONFIGURATIONS = {
    "verification": ("render_verification", [LOOKUP_CUSTOMER_TOOL]),
    "policy": ("render_policy", [SEARCH_POLICY_TOOL]),
    "intake": ("render_intake", [RECORD_INTAKE_TOOL, REQUEST_HANDOFF_TOOL]),
    "scheduling": ("render_scheduling", [FIND_APPOINTMENT_SLOTS_TOOL, SELECT_APPOINTMENT_SLOT_TOOL, BOOK_SELECTED_APPOINTMENT_TOOL]),
    "human_handoff": ("render_human_handoff", [REQUEST_HANDOFF_TOOL]),
}


def speech(text="Final caller-facing fixture text."):
    return SimpleNamespace(output=[], output_text=text)


def tool_call(name, call_id="controlled-call"):
    return SimpleNamespace(type="function_call", name=name, call_id=call_id, arguments="{}")


def call_response(name):
    return SimpleNamespace(output=[tool_call(name)], output_text="")


def judgment(case, *, passed=True):
    return ResponseJudgeResult(
        passed=passed, reason="Fixture criteria pass." if passed else "The fixture response violates a criterion.",
        criteria_results=[
            {"criterion_id": criterion.criterion_id, "passed": passed or index > 0}
            for index, criterion in enumerate(case.criteria)
        ],
    )


class FakeResponses:
    def __init__(self):
        self.outputs = []
        self.judgments = []
        self.calls = []
        self.judge_calls = []
        self.events = []
        self.active = 0
        self.max_active = 0

    async def create(self, **kwargs):
        self.calls.append({**kwargs, "input": list(kwargs["input"])})
        self.events.append("specialist")
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            await asyncio.sleep(0)
            assert self.outputs, "No additional specialist call was expected"
            result = self.outputs.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            self.active -= 1

    async def parse(self, **kwargs):
        self.judge_calls.append(kwargs)
        self.events.append("judge")
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            await asyncio.sleep(0)
            assert self.judgments, "No judge call was expected"
            result = self.judgments.pop(0)
            if isinstance(result, Exception):
                raise result
            return SimpleNamespace(output_parsed=result)
        finally:
            self.active -= 1


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *args):
        self.closed = True


@pytest.fixture(autouse=True)
def forbid_network_and_production_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live adapter tests must use fake OpenAI only; no production state, handlers, or integrations")

    monkeypatch.setattr(AsyncOpenAI, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for cls in (LegacyCRMClient, SchedulerClient, HandoffClient, PolicyRetriever, ConversationState, InMemoryConversationStore):
        monkeypatch.setattr(cls, "__init__", forbidden)
    for cls, methods in (
        (LegacyCRMClient, ["find_customer_by_phone"]), (SchedulerClient, ["find_slots", "book_appointment"]),
        (HandoffClient, ["create_handoff"]), (PolicyRetriever, ["search"]),
        (InMemoryConversationStore, ["create", "get", "save"]),
    ):
        for name in methods:
            monkeypatch.setattr(cls, name, forbidden)
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
    monkeypatch.setattr(import_module("app.agents.responses_tool_loop"), "run_agent_turn", forbidden)
    for module_name, class_name in (
        ("verification_agent", "VerificationAgent"), ("policy_agent", "PolicyAgent"),
        ("intake_agent", "IntakeAgent"), ("scheduling_agent", "SchedulingAgent"),
        ("human_handoff_agent", "HumanHandoffAgent"),
    ):
        module = import_module("app.agents." + module_name)
        monkeypatch.setattr(getattr(module, class_name), "__init__", forbidden)
        monkeypatch.setattr(module, "run_agent_turn", forbidden)


@pytest.fixture
def cli_setup(monkeypatch):
    client = FakeOpenAIClient()
    factory = Mock(return_value=client)
    monkeypatch.setattr(live_response_eval, "AsyncOpenAI", factory)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("OPENAI_SPECIALIST_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EVAL_JUDGE_MODEL", raising=False)
    return client, factory


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
async def test_tool_case_makes_one_continuation_with_completed_call_and_result(case, monkeypatch):
    client = FakeOpenAIClient()
    client.responses.outputs = [speech()]
    before = deepcopy(case)
    renderer = PromptRenderer()
    method, tools = CONFIGURATIONS[case.specialist]
    render = Mock(wraps=getattr(renderer, method))
    monkeypatch.setattr(renderer, method, render)
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "specialist-model", renderer)

    assert await generate(case) == "Final caller-facing fixture text."

    assert case == before
    assert len(client.responses.calls) == 1
    render.assert_called_once()
    safe_context = PromptContext.model_validate({"active_agent": case.specialist, **live_response_eval.build_response_context(case)})
    for request in client.responses.calls:
        assert request["instructions"] == getattr(PromptRenderer(), method)(safe_context)
        assert request["tools"] == tools
        assert all(actual is expected for actual, expected in zip(request["tools"], tools, strict=True))
        assert request["model"] == "specialist-model"
        assert request["parallel_tool_calls"] is False
    inputs = client.responses.calls[0]["input"]
    call_id = f"call_response_eval_{case.case_id}"
    assert inputs == [
        {"role": "user", "content": case.user_message},
        {
            "type": "function_call", "call_id": call_id, "name": case.tool_name,
            "arguments": json.dumps(case.tool_arguments, ensure_ascii=False, allow_nan=False),
        },
        {
            "type": "function_call_output", "call_id": call_id,
            "output": json.dumps(case.tool_result, ensure_ascii=False, allow_nan=False),
        },
    ]
    output = inputs[-1]
    assert json.loads(output["output"]) == case.tool_result
    assert inputs[1]["call_id"] == output["call_id"]
    assert client.responses.judge_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["record_intake", "request_handoff", "unknown_tool"])
async def test_another_tool_request_is_rejected_without_execution_or_another_request(tool):
    case = next(case for case in CASES if case.specialist == "intake")
    client = FakeOpenAIClient()
    client.responses.outputs = [call_response(tool)]
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer())
    with pytest.raises(live_response_eval.LiveResponseEvaluationError, match="requested a tool instead of returning final speech"):
        await generate(case)
    assert len(client.responses.calls) == 1


@pytest.mark.asyncio
async def test_tool_request_is_rejected_even_if_partial_speech_is_also_present():
    client = FakeOpenAIClient()
    response = call_response("lookup_customer")
    response.output_text = "Partial response alongside a tool request."
    client.responses.outputs = [response]
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer())
    with pytest.raises(live_response_eval.LiveResponseEvaluationError, match="requested a tool"):
        await generate(CASES[0])
    assert len(client.responses.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [None, SimpleNamespace(output_text="Text without output items"), SimpleNamespace(output=None)])
async def test_missing_output_list_is_rejected_safely(response):
    client = FakeOpenAIClient()
    client.responses.outputs = [response]
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer())
    with pytest.raises(live_response_eval.LiveResponseEvaluationError, match="missing an output list"):
        await generate(CASES[0])


@pytest.mark.asyncio
async def test_argument_validation_does_not_rewrite_serialized_values():
    arguments = {"phone": " 202-555-0142 "}
    case = CASES[0].model_copy(update={"tool_arguments": arguments})
    client = FakeOpenAIClient()
    client.responses.outputs = [speech()]
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer())
    await generate(case)
    assert client.responses.calls[0]["input"][1]["arguments"] == json.dumps(arguments, ensure_ascii=False, allow_nan=False)
    assert case.tool_arguments == arguments


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_tool", [False, True])
async def test_speech_only_cases_send_only_user_input_and_reject_tools(requests_tool):
    case = CASES[0].model_copy(update={"tool_name": None, "tool_arguments": None, "tool_result": None, "user_message": "What phone number do you need?"})
    client = FakeOpenAIClient()
    client.responses.outputs = [call_response("lookup_customer") if requests_tool else speech("What is your phone number?")]
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer())
    if requests_tool:
        with pytest.raises(live_response_eval.LiveResponseEvaluationError, match="requested a tool instead of returning final speech"):
            await generate(case)
    else:
        assert await generate(case) == "What is your phone number?"
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["input"] == [{"role": "user", "content": case.user_message}]


@pytest.mark.asyncio
async def test_speech_only_pipeline_makes_one_specialist_call_and_one_judge_call():
    case = CASES[0].model_copy(update={
        "tool_name": None, "tool_arguments": None, "tool_result": None, "user_message": "Can you find my account?",
        "criteria": [ResponseCriterion(criterion_id="ask_phone", description="Ask for the caller's phone number without claiming verification.")],
    })
    client = FakeOpenAIClient()
    final = "What phone number should I use to find your account?"
    client.responses.outputs = [speech(final)]
    client.responses.judgments = [judgment(case)]

    report = await run_response_evaluation(
        [case], live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer()),
        live_response_eval.create_live_grade_response(cast(AsyncOpenAI, client), "judge"),
    )

    assert report["passed"] == 1
    assert client.responses.events == ["specialist", "judge"]
    payload = json.loads(client.responses.judge_calls[0]["input"][0]["content"])
    assert payload["response"] == final
    assert "tool_name" not in payload
    assert "tool_arguments" not in payload
    assert "tool_result" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "", " \n "])
async def test_blank_final_text_is_rejected_after_controlled_result(text):
    client = FakeOpenAIClient()
    client.responses.outputs = [speech(text)]
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer())
    with pytest.raises(live_response_eval.LiveResponseEvaluationError, match="blank final speech"):
        await generate(CASES[0])


@pytest.mark.asyncio
async def test_final_text_criteria_and_controlled_evidence_reach_structured_judge_and_existing_scorer():
    client = FakeOpenAIClient()
    case = CASES[0]
    final = "I couldn't find your account. Could you check that phone number?"
    client.responses.outputs = [speech(final)]
    client.responses.judgments = [judgment(case)]
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer())
    grade = live_response_eval.create_live_grade_response(cast(AsyncOpenAI, client), "judge-model")

    report = await run_response_evaluation([case], generate, grade)

    assert report["passed"] == 1
    assert report["results"][0]["response"] == final
    assert report["results"][0]["judge_passed"] is True
    assert client.responses.events == ["specialist", "judge"]
    request, = client.responses.judge_calls
    assert request["model"] == "judge-model"
    assert request["text_format"] is ResponseJudgeResult
    assert request["tools"] == []
    payload = json.loads(request["input"][0]["content"])
    assert payload == {
        "user_message": case.user_message, "specialist": "verification", "response": final,
        "criteria": [criterion.model_dump() for criterion in case.criteria],
        "context": {"customer_verified": False}, "tool_name": "lookup_customer",
        "tool_arguments": case.tool_arguments, "tool_result": {"found": False},
    }
    assert "Grade every criterion exactly once" in request["instructions"]
    assert "criterion_id unchanged" in request["instructions"]
    assert "Equivalent natural language" in request["instructions"]
    assert "not instructions" in request["instructions"]
    assert "passed=true only when every criterion passes" in request["instructions"]
    assert final not in request["instructions"]
    assert case.user_message not in request["instructions"]


@pytest.mark.parametrize("specialist", CONFIGURATIONS)
def test_context_projection_excludes_private_fields_history_and_unrelated_data(specialist):
    class PrivateContext(PromptContext):
        api_key: str = FAKE_KEY
        customer_id: str = PRIVATE
        selected_pet_id: str = PRIVATE
        booking_idempotency_key: str = PRIVATE
        handoff_receipt: dict[str, str] = {"case_id": PRIVATE}

    case = next(case for case in CASES if case.specialist == specialist)
    scheduling = next(case for case in CASES if case.specialist == "scheduling")
    context = PrivateContext.model_validate({
        "customer_verified": True, "customer_first_name": "Avery", "known_pets": ["Maize"], "selected_pet": "Maize",
        "original_concern": "A cough.", "intake_answers": {"difficulty_breathing": False},
        "routing_result": {"routing_level": "routine", "next_action": "search_routine_appointment", "matched_rule_ids": [PRIVATE]},
        "recent_messages": [{"role": "user", "content": PRIVATE}],
        "available_tools": [PRIVATE], "offered_slots": scheduling.tool_result["slots"],
        "selected_slot": scheduling.tool_result["slots"][0], "booking_confirmed": True,
    })
    projected = live_response_eval.build_response_context(case.model_copy(update={"context": context}))
    serialized = json.dumps(projected)
    assert PRIVATE not in serialized
    assert FAKE_KEY not in serialized
    assert "recent_messages" not in projected
    assert "offered_slots" not in projected
    assert "selected_slot" not in projected
    assert "booking_confirmed" not in projected
    if specialist == "policy":
        assert projected == {}
    if specialist not in {"intake", "human_handoff"}:
        assert "intake_answers" not in projected
        assert "original_concern" not in projected


@pytest.mark.asyncio
async def test_private_context_is_excluded_from_both_specialist_and_judge_requests():
    case = next(case for case in CASES if case.specialist == "intake")
    context = case.context.model_copy(update={"recent_messages": [], "available_tools": [PRIVATE]})
    # Caller history is not sent, even though the original fixture includes a prior question.
    case = case.model_copy(update={"context": context})
    client = FakeOpenAIClient()
    client.responses.outputs = [speech()]
    client.responses.judgments = [judgment(case)]
    await run_response_evaluation(
        [case], live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer()),
        live_response_eval.create_live_grade_response(cast(AsyncOpenAI, client), "judge"),
    )
    for request in client.responses.calls:
        assert PRIVATE not in request["instructions"]
        assert "assistant: Is Maize having difficulty breathing?" not in request["instructions"]
    assert PRIVATE not in client.responses.judge_calls[0]["input"][0]["content"]


@pytest.mark.parametrize("key", ["api_key", "customer_id", "booking_idempotency_key", "raw_response", "exception_details", "pet_id"])
def test_forbidden_controlled_fixture_fields_abort_before_creating_a_client(cli_setup, monkeypatch, key):
    client, factory = cli_setup
    bad = CASES[0].model_copy(update={"tool_result": {"found": False, "nested": {key: FAKE_KEY}}})
    monkeypatch.setattr(live_response_eval, "load_response_cases", lambda: [bad])
    assert live_response_eval.main(["--confirm-live", "--case-id", bad.case_id]) == 1
    factory.assert_not_called()
    assert client.responses.calls == []


@pytest.mark.parametrize("arguments", [{}, {"phone": "202-555-0142", "customer_id": PRIVATE}, {"phone": FAKE_KEY}])
def test_invalid_or_private_arguments_fail_before_any_network_request(cli_setup, monkeypatch, capsys, arguments):
    client, factory = cli_setup
    bad = CASES[0].model_copy(update={"tool_arguments": arguments})
    monkeypatch.setattr(live_response_eval, "load_response_cases", lambda: [bad])
    assert live_response_eval.main(["--confirm-live", "--case-id", bad.case_id]) == 1
    factory.assert_not_called()
    assert client.responses.calls == []
    output = capsys.readouterr()
    assert PRIVATE not in output.err
    assert FAKE_KEY not in output.err


def test_explicit_pet_id_criterion_is_required_for_pet_id_fixture_values():
    case = CASES[0].model_copy(update={
        "tool_result": {"found": True, "pets": [{"pet_id": "fictional-pet-id", "name": "Maize"}]},
        "criteria": [ResponseCriterion(criterion_id="exact_pet_id", description="Include the exact fictional pet ID from the result.")],
    })
    live_response_eval._validate_controlled_fixture(case)


@pytest.mark.asyncio
async def test_api_key_in_otherwise_allowed_data_is_rejected_before_a_request(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    case = CASES[0].model_copy(update={"user_message": FAKE_KEY})
    client = FakeOpenAIClient()
    generate = live_response_eval.create_live_generate_response(cast(AsyncOpenAI, client), "model", PromptRenderer())
    grade = live_response_eval.create_live_grade_response(cast(AsyncOpenAI, client), "judge")
    with pytest.raises(live_response_eval.LiveResponseEvaluationError, match="Configured credentials"):
        await generate(case)
    with pytest.raises(live_response_eval.LiveResponseEvaluationError, match="Configured credentials"):
        await grade(case, "Final text.")
    assert client.responses.calls == client.responses.judge_calls == []


def test_cli_requires_confirmation_before_client_creation(cli_setup, capsys):
    _, factory = cli_setup
    with pytest.raises(SystemExit) as caught:
        live_response_eval.main(["--case-id", SMOKE_CASE])
    assert caught.value.code == 2
    assert "--confirm-live" in capsys.readouterr().err
    factory.assert_not_called()


@pytest.mark.parametrize("key", [None, "", " \t "])
def test_cli_requires_api_key_before_client_creation(cli_setup, monkeypatch, key):
    _, factory = cli_setup
    if key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    with pytest.raises(SystemExit) as caught:
        live_response_eval.main(["--confirm-live", "--case-id", SMOKE_CASE])
    assert caught.value.code == 2
    factory.assert_not_called()


@pytest.mark.parametrize("selection", [[], ["--all-cases", "--case-id", SMOKE_CASE], ["--case-id", "unknown-case"], ["--case-id", SMOKE_CASE, "--case-id", "unknown-case"]])
def test_cli_requires_exclusive_valid_case_selection(cli_setup, capsys, selection):
    _, factory = cli_setup
    with pytest.raises(SystemExit) as caught:
        live_response_eval.main(["--confirm-live", *selection])
    assert caught.value.code == 2
    if "unknown-case" in selection:
        assert "Unknown case IDs: unknown-case" in capsys.readouterr().err
    factory.assert_not_called()


@pytest.mark.parametrize("all_cases", [False, True])
def test_cli_selection_runs_only_requested_cases_sequentially_with_existing_scorer(cli_setup, monkeypatch, capsys, all_cases):
    client, factory = cli_setup
    selected = CASES if all_cases else CASES[:1]
    client.responses.outputs = [speech() for case in selected]
    client.responses.judgments = [judgment(case) for case in selected]
    runner = Mock(wraps=run_response_evaluation)
    monkeypatch.setattr(live_response_eval, "run_response_evaluation", runner)
    args = ["--confirm-live", "--all-cases"] if all_cases else ["--confirm-live", "--case-id", SMOKE_CASE, "--case-id", SMOKE_CASE]

    assert live_response_eval.main(args) == 0

    runner.assert_called_once()
    assert runner.call_args.args[0] == selected
    factory.assert_called_once_with(api_key=FAKE_KEY)
    assert client.entered and client.closed
    assert len(client.responses.calls) == len(selected)
    assert len(client.responses.judge_calls) == len(selected)
    assert client.responses.events == ["specialist", "judge"] * len(selected)
    assert client.responses.max_active == 1
    output = capsys.readouterr()
    assert f"Total: {len(selected)}\nPassed: {len(selected)}\nFailed: 0\nPass rate: 100.0%" in output.out
    assert output.err == ""
    assert FAKE_KEY not in output.out


@pytest.mark.parametrize("cli_model, env_model, judge_model, expected", [
    ("cli-model", "env-model", "judge-model", "cli-model"),
    (None, "env-model", None, "env-model"),
    (None, None, None, "gpt-6-luna"),
])
def test_model_configuration_and_judge_fallback(cli_setup, monkeypatch, cli_model, env_model, judge_model, expected):
    client, _ = cli_setup
    if env_model:
        monkeypatch.setenv("OPENAI_SPECIALIST_MODEL", env_model)
    if judge_model:
        monkeypatch.setenv("OPENAI_EVAL_JUDGE_MODEL", judge_model)
    client.responses.outputs = [speech()]
    client.responses.judgments = [judgment(CASES[0])]
    args = ["--confirm-live", "--case-id", SMOKE_CASE]
    if cli_model:
        args.extend(["--model", cli_model])
    assert live_response_eval.main(args) == 0
    assert all(call["model"] == expected for call in client.responses.calls)
    assert client.responses.judge_calls[0]["model"] == (judge_model or expected)


@pytest.mark.parametrize("setting", ["cli", "OPENAI_SPECIALIST_MODEL", "OPENAI_EVAL_JUDGE_MODEL"])
def test_blank_model_configuration_is_rejected_before_client_creation(cli_setup, monkeypatch, setting):
    _, factory = cli_setup
    args = ["--confirm-live", "--case-id", SMOKE_CASE]
    if setting == "cli":
        args.extend(["--model", "  "])
    else:
        monkeypatch.setenv(setting, "  ")
    with pytest.raises(SystemExit) as caught:
        live_response_eval.main(args)
    assert caught.value.code == 2
    factory.assert_not_called()


def test_failed_judgment_returns_exit_one_and_reports_failed_criteria(cli_setup, capsys):
    client, _ = cli_setup
    client.responses.outputs = [speech()]
    client.responses.judgments = [judgment(CASES[0], passed=False)]
    assert live_response_eval.main(["--confirm-live", "--case-id", SMOKE_CASE]) == 1
    output = capsys.readouterr()
    assert "Passed: 0\nFailed: 1" in output.out
    assert "acknowledge_not_found" in output.out


def test_api_key_in_a_structured_judge_reason_is_not_printed(cli_setup, capsys):
    client, _ = cli_setup
    client.responses.outputs = [speech()]
    client.responses.judgments = [judgment(CASES[0]).model_copy(update={"reason": FAKE_KEY})]
    assert live_response_eval.main(["--confirm-live", "--case-id", SMOKE_CASE]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert FAKE_KEY not in output.err


@pytest.mark.parametrize("stage", ["specialist", "judge", "malformed_judge", "client_init"])
def test_upstream_errors_never_leak_prompts_credentials_or_responses(cli_setup, capsys, stage):
    client, factory = cli_setup
    error = RuntimeError(f"{FAKE_KEY} {PRIVATE} raw-upstream-response")
    client.responses.outputs = [speech()]
    if stage == "specialist":
        client.responses.outputs = [error]
    elif stage == "judge":
        client.responses.judgments = [error]
    elif stage == "malformed_judge":
        client.responses.judgments = [{"passed": True, "reason": str(error)}]
    else:
        factory.side_effect = error

    assert live_response_eval.main(["--confirm-live", "--case-id", SMOKE_CASE]) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert "Live response evaluation failed" in output.err
    assert FAKE_KEY not in output.err
    assert PRIVATE not in output.err
    assert "raw-upstream-response" not in output.err
    if stage != "client_init":
        assert client.closed
