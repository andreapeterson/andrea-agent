"""Run real conversation code with scripted OpenAI responses and demo integrations."""

import asyncio
import json
import socket
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from app.agents.policy_agent import PolicyAgent
from app.agents.prompt_renderer import PromptRenderer
from app.agents.router_agent import RouteDecision
from app.agents.responses_tool_loop import MalformedToolCallError, ToolLoopLimitError, UnknownAgentToolError, run_agent_turn
from app.agents.responses_tool_loop import AgentModelRequestError, ToolExecutionError
from app.agents.prompt_renderer import POLICY_PROMPT_VERSION, ROUTER_PROMPT_VERSION
from app.agents.router_agent import RouterRequestError
from app.agents.search_policy_tool import create_search_policy_handler
from app.agents.workflow_observation import WorkflowEventType, observe_specialist, observe_tool, observe_turn
from app.integrations import HandoffClient, LegacyCRMClient, SchedulerClient
from app.models.appointment import AppointmentType, BookingRequest
from app.models.conversation import ConversationPhase
from app.models.handoff import HandoffRequest
from app.services.conversation_store import InMemoryConversationStore
from app.services.policy_retriever import PolicyRetriever
from app.services.prompt_agent_runtime import PromptAgentRuntime
from evals import live_conversation_eval
from evals.conversation_eval import (
    ConversationEvalCase,
    ConversationRunObservation,
    load_conversation_cases,
    run_conversation_evaluation,
)
from evals.live_conversation_eval import (
    ControlledIntegrationError,
    DemoCRM,
    DemoHandoff,
    DemoPolicyRetriever,
    DemoScheduler,
    LiveConversationEvaluationError,
    create_live_run_conversation,
)


def speech(text="Controlled caller-facing reply."):
    return SimpleNamespace(output=[], output_text=text)


def tool(name, arguments):
    return SimpleNamespace(output=[SimpleNamespace(
        type="function_call", call_id=f"call-{name}", name=name, arguments=json.dumps(arguments),
    )], output_text="")


def intake_arguments(*, urgent):
    return {
        "original_concern": "Maize is having difficulty breathing." if urgent else "Maize has a mild cough.",
        "difficulty_breathing": urgent, "uncontrolled_bleeding": None if urgent else False,
        "collapsed_or_unresponsive": None if urgent else False,
        "known_toxin_exposure": None if urgent else False,
        "rapidly_worsening": None if urgent else False,
    }


class FakeResponses:
    def __init__(self, routes, outputs):
        self.routes = list(routes)
        self.outputs = list(outputs)
        self.parse_calls = []
        self.create_calls = []
        self.order = []

    async def parse(self, **request):
        self.parse_calls.append(deepcopy(request))
        self.order.append("router")
        await asyncio.sleep(0)
        result = self.routes.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(output_parsed=RouteDecision(destination=result, reason="Scripted route."))

    async def create(self, **request):
        self.create_calls.append(deepcopy(request))
        self.order.append("specialist")
        await asyncio.sleep(0)
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeOpenAI:
    def __init__(self, routes, outputs):
        self.responses = FakeResponses(routes, outputs)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


def script(index):
    if index == 0:
        return (["policy"], [tool("search_policy", {"query": "What are your clinic hours?"}), speech()])
    if index == 1:
        return (["verification", "intake"], [
            tool("lookup_customer", {"phone": "202-555-0142"}), speech(),
            tool("record_intake", intake_arguments(urgent=True)),
            tool("request_handoff", {"reason": "urgent_intake", "explanation": "Recorded intake reports difficulty breathing."}), speech(),
        ])
    return (["verification", "intake", "scheduling", "scheduling", "scheduling"], [
        tool("lookup_customer", {"phone": "202-555-0143"}), speech(),
        tool("record_intake", intake_arguments(urgent=False)), speech(),
        tool("find_appointment_slots", {"pet_reference": "Maize"}), speech(),
        tool("select_appointment_slot", {"slot_id": "demo-routine-slot-1"}), speech(),
        tool("book_selected_appointment", {"confirmed_by_caller": True}), speech(),
    ])


@pytest.fixture(autouse=True)
def forbid_real_clients_and_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests may use fake OpenAI clients and controlled integrations only")

    for name in ("OPENAI_API_KEY", "OPENAI_ROUTER_MODEL", "OPENAI_SPECIALIST_MODEL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for cls in (AsyncOpenAI, OpenAI, LegacyCRMClient, SchedulerClient, HandoffClient, PolicyRetriever, httpx.Client, httpx.AsyncClient):
        monkeypatch.setattr(cls, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def runner(client):
    return create_live_run_conversation(client, "demo-router-model", "demo-specialist-model", PromptRenderer())


def selected_workflow(events):
    return [(event.conversation_id, event.turn_number, event.specialist, event.tool_name)
            for event in events if event.event_type in {
                WorkflowEventType.SPECIALIST_SELECTED, WorkflowEventType.TOOL_REQUESTED,
            }]


def build_runtime(client, observer=None):
    store = InMemoryConversationStore()
    renderer = PromptRenderer()
    runtime = PromptAgentRuntime(
        conversation_store=store, openai_client=client,
        router_model="demo-router-model", specialist_model="demo-specialist-model",
        prompt_renderer=renderer, legacy_crm_client=DemoCRM(), scheduler_client=DemoScheduler(),
        handoff_client=DemoHandoff(), idempotency_key_factory=lambda: "demo-test-key",
        policy_agent=PolicyAgent(
            client=client, model="demo-specialist-model", prompt_renderer=renderer,
            search_policy_handler=create_search_policy_handler(DemoPolicyRetriever()),
        ), workflow_observer=observer,
    )
    return runtime, store


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [0, 1, 2])
async def test_real_runtime_workflows_produce_observations_for_existing_scorer(index):
    case = load_conversation_cases()[index]
    client = FakeOpenAI(*script(index))
    observation = await runner(client)(case)

    assert isinstance(observation, ConversationRunObservation)
    assert observation.specialists_by_turn == case.expected_specialists_by_turn
    assert observation.tools_by_turn == case.expected_tools_by_turn
    assert observation.final_phase == case.expected_final_phase
    assert observation.verified_customer == case.expected_verified_customer
    assert observation.booking_confirmed == case.expected_booking_confirmed
    assert observation.handoff_queued == case.expected_handoff_queued
    assert [request["input"] for request in client.responses.parse_calls] == case.caller_turns
    assert all(request["model"] == "demo-router-model" for request in client.responses.parse_calls)
    assert all(request["model"] == "demo-specialist-model" for request in client.responses.create_calls)
    assert not client.responses.routes and not client.responses.outputs
    report = await run_conversation_evaluation([case], AsyncMock(return_value=observation))
    assert report["passed"] == 1 and report["failures"] == []


@pytest.mark.asyncio
async def test_state_survives_each_sequential_turn_and_tool_results_return_to_model():
    case = load_conversation_cases()[2]
    client = FakeOpenAI(*script(2))
    observation = await runner(client)(case)
    routes = client.responses.parse_calls
    assert "Customer verified: false" in routes[0]["instructions"]
    assert "Customer verified: true" in routes[1]["instructions"]
    assert case.caller_turns[0] in routes[1]["instructions"]
    assert "routine / search_routine_appointment" in routes[2]["instructions"]
    assert "Offered slot IDs: demo-routine-slot-1, demo-routine-slot-2" in routes[3]["instructions"]
    assert "Selected slot ID: demo-routine-slot-1" in routes[4]["instructions"]
    outputs = [
        item for request in client.responses.create_calls
        for item in request["input"] if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    payloads = [json.loads(item["output"]) for item in outputs]
    assert [payload.get("status", "recorded" if "intake_answers" in payload else "found") for payload in payloads] == [
        "found", "recorded", "slots_found", "slot_selected", "booking_confirmed",
    ]
    assert payloads[0]["found"] is True
    assert payloads[2]["selected_pet"]["name"] == "Maize"
    assert payloads[4]["confirmation"]["slot_id"] == "demo-routine-slot-1"
    assert observation.booking_confirmed is True
    assert client.responses.order == ["router", "specialist", "specialist"] * 5


@pytest.mark.asyncio
async def test_expected_labels_are_never_model_inputs_or_runtime_context():
    original = load_conversation_cases()[2]
    payload = original.model_dump(mode="json")
    payload.update(
        case_id="SECRET_ANSWER_KEY_CASE", tags=["SECRET_LABEL_TAG"],
        expected_specialists_by_turn=["policy"] * len(original.caller_turns),
        expected_tools_by_turn=[[] for _ in original.caller_turns],
        expected_final_phase="verifying_customer", expected_verified_customer=False,
        expected_booking_confirmed=False, expected_handoff_queued=False,
    )
    misleading_case = ConversationEvalCase.model_validate(payload)
    client = FakeOpenAI(*script(2))
    observation = await runner(client)(misleading_case)
    assert observation.specialists_by_turn == original.expected_specialists_by_turn
    assert observation.tools_by_turn == original.expected_tools_by_turn
    assert observation.booking_confirmed is True
    requests = repr(client.responses.parse_calls + client.responses.create_calls)
    for private in ("SECRET_ANSWER_KEY_CASE", "SECRET_LABEL_TAG", *[key for key in payload if key.startswith("expected_")]):
        assert private not in requests
    assert [request["input"] for request in client.responses.parse_calls] == original.caller_turns


@pytest.mark.asyncio
async def test_wrong_specialist_and_missing_tools_are_scored_without_rewriting():
    case = load_conversation_cases()[0]
    client = FakeOpenAI(["verification"], [speech("Booking confirmed; a live transfer is connected.")])
    report = await run_conversation_evaluation([case], runner(client))
    result = report["results"][0]
    assert result["actual"]["specialist_sequence"] == ["verification"]
    assert result["actual"]["tool_sequence"] == [[]]
    assert not result["actual"]["verified_customer"]
    assert not result["actual"]["booking_confirmed"]
    assert not result["actual"]["handoff_queued"]
    assert not result["checks"]["specialist_sequence"] and not result["checks"]["tool_sequence"]


@pytest.mark.asyncio
async def test_additional_approved_tool_calls_are_scored_as_observed():
    case = load_conversation_cases()[0]
    client = FakeOpenAI(["policy"], [
        tool("search_policy", {"query": "clinic hours"}),
        tool("search_policy", {"query": "opening hours"}), speech(),
    ])
    report = await run_conversation_evaluation([case], runner(client))
    result = report["results"][0]
    assert result["actual"]["tool_sequence"] == [["search_policy", "search_policy"]]
    assert result["checks"]["specialist_sequence"] and not result["checks"]["tool_sequence"]


@pytest.mark.asyncio
async def test_urgent_tool_claim_does_not_create_trusted_handoff_outcome():
    routes, outputs = script(1)
    outputs[2] = tool("record_intake", intake_arguments(urgent=False))
    client = FakeOpenAI(routes, outputs)
    observation = await runner(client)(load_conversation_cases()[1])
    assert observation.tools_by_turn[1] == ["record_intake", "request_handoff"]
    assert observation.verified_customer and not observation.handoff_queued
    assert observation.final_phase == ConversationPhase.VERIFYING_CUSTOMER
    result = json.loads(client.responses.create_calls[-1]["input"][-1]["output"])
    assert result == {"queued": False, "case_id": None, "error": "handoff-not-required"}


@pytest.mark.asyncio
async def test_hook_records_only_allowlisted_fields_and_preserves_tool_order():
    client = FakeOpenAI(*script(1))
    events = []
    runtime, store = build_runtime(client, events.append)
    case = load_conversation_cases()[1]
    await runtime.start_conversation("demo-observed")
    for message in case.caller_turns:
        await runtime.handle_turn("demo-observed", message)

    assert selected_workflow(events) == [
        ("demo-observed", 1, "verification", None),
        ("demo-observed", 1, "verification", "lookup_customer"),
        ("demo-observed", 2, "intake", None),
        ("demo-observed", 2, "intake", "record_intake"),
        ("demo-observed", 2, "intake", "request_handoff"),
    ]
    assert all(set(asdict(event)) == {
        "conversation_id", "turn_number", "event_type", "specialist", "tool_name",
        "prompt_version", "duration_ms", "model", "status",
    } for event in events)
    assert "Maize" not in repr(events) and "202-555" not in repr(events)
    state = store.get("demo-observed")
    assert state.turn_count == 2
    assert state.verified_customer is not None and state.handoff_receipt is not None
    assert state.booking_confirmation is None
    assert state.phase == ConversationPhase.HANDOFF_COMPLETE


@pytest.mark.asyncio
async def test_optional_hook_does_not_change_model_requests_state_or_replies():
    clients = [FakeOpenAI(*script(0)), FakeOpenAI(*script(0))]
    events = []
    results = []
    for client, observer in zip(clients, (None, events.append), strict=True):
        runtime, store = build_runtime(client, observer)
        await runtime.start_conversation("demo-optional")
        result = await runtime.handle_turn("demo-optional", "What are your clinic hours?")
        results.append((result, store.get("demo-optional")))
    assert results[0] == results[1]
    assert clients[0].responses.parse_calls == clients[1].responses.parse_calls
    assert clients[0].responses.create_calls == clients[1].responses.create_calls
    assert sum(event.event_type == WorkflowEventType.TURN_COMPLETED for event in events) == 1
    assert len(selected_workflow(events)) == 2


@pytest.mark.asyncio
async def test_trace_lifecycle_has_request_metadata_and_completes_after_state_save():
    events = []
    client = FakeOpenAI(*script(0))
    runtime, store = build_runtime(client, events.append)
    await runtime.start_conversation("demo-lifecycle")
    assert events == []  # The initial greeting is not a caller turn.
    save = store.save

    def save_before_turn_completion(state):
        assert events[-1].event_type == WorkflowEventType.MODEL_REQUEST_COMPLETED
        return save(state)

    store.save = save_before_turn_completion
    await runtime.handle_turn("demo-lifecycle", "What are your clinic hours?")
    assert [event.event_type.value for event in events] == [
        "turn_started", "model_request_started", "model_request_completed",
        "specialist_selected", "model_request_started", "model_request_completed",
        "tool_requested", "tool_completed", "model_request_started", "model_request_completed",
        "turn_completed",
    ]
    assert {(event.conversation_id, event.turn_number) for event in events} == {("demo-lifecycle", 1)}
    requests = [event for event in events if event.event_type == WorkflowEventType.MODEL_REQUEST_STARTED]
    assert [(event.specialist, event.model, event.prompt_version) for event in requests] == [
        (None, "demo-router-model", ROUTER_PROMPT_VERSION),
        ("policy", "demo-specialist-model", POLICY_PROMPT_VERSION),
        ("policy", "demo-specialist-model", POLICY_PROMPT_VERSION),
    ]
    for event in events:
        if event.event_type in {
            WorkflowEventType.MODEL_REQUEST_COMPLETED, WorkflowEventType.TOOL_COMPLETED, WorkflowEventType.TURN_COMPLETED,
        }:
            assert event.duration_ms >= 0 and event.status == "succeeded"
        else:
            assert event.duration_ms is None and event.status is None
    assert store.get("demo-lifecycle").turn_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_event", [kind for kind in WorkflowEventType if kind != WorkflowEventType.TURN_FAILED])
async def test_observer_failure_cannot_change_caller_reply_state_or_requests(broken_event, capsys):
    events = []

    def broken_observer(event):
        events.append(event)
        if event.event_type == broken_event:
            raise RuntimeError("PRIVATE_OBSERVER_EXCEPTION and secret credentials")

    clients = [FakeOpenAI(*script(0)), FakeOpenAI(*script(0))]
    results = []
    for client, observer in zip(clients, (None, broken_observer), strict=True):
        runtime, store = build_runtime(client, observer)
        await runtime.start_conversation("demo-observer-failure")
        result = await runtime.handle_turn("demo-observer-failure", "What are your clinic hours?")
        results.append((result, store.get("demo-observer-failure")))
    assert results[0] == results[1]
    assert clients[0].responses.parse_calls == clients[1].responses.parse_calls
    assert clients[0].responses.create_calls == clients[1].responses.create_calls
    assert events[-1].event_type == WorkflowEventType.TURN_COMPLETED
    output = capsys.readouterr()
    assert output.out == output.err == ""


@pytest.mark.asyncio
async def test_trace_contains_no_private_content_and_is_never_sent_to_model(monkeypatch):
    events = []
    private = "PRIVATE_CALLER_MESSAGE_PRIVATE_RESULT"
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_API_KEY_VALUE")
    client = FakeOpenAI(["verification"], [tool("lookup_customer", {"phone": "202-555-0142"}), speech(private)])
    runtime, _ = build_runtime(client, events.append)
    await runtime.start_conversation("demo-private")
    await runtime.handle_turn("demo-private", private + " My phone number is 202-555-0142.")
    serialized = json.dumps([asdict(event) for event in events])
    for forbidden in (
        private, "PRIVATE_API_KEY_VALUE", "202-555-0142", "Avery", "Maize",
        "demo-customer-0142", "demo-pet-maize-0142", "call-lookup_customer",
        "Identify a caller", "authorization", "arguments", "output_text", "traceback",
    ):
        assert forbidden not in serialized
    assert "Customer verified:" not in serialized
    for request in client.responses.parse_calls + client.responses.create_calls:
        assert "workflow_observer" not in request
        assert "events" not in request
        assert "turn_started" not in repr(request)
        assert "tool_completed" not in repr(request)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage, error_type", [
    ("router", RouterRequestError), ("model", AgentModelRequestError),
    ("tool", ToolExecutionError), ("save", RuntimeError),
])
async def test_failures_emit_safe_status_without_masking_workflow_error(stage, error_type):
    events = []
    private_error = "PRIVATE_EXCEPTION_DETAIL tool arguments phone=202-555-0142 secret-key"
    if stage == "router":
        client = FakeOpenAI([RuntimeError(private_error)], [])
    elif stage == "model":
        client = FakeOpenAI(["policy"], [RuntimeError(private_error)])
    elif stage == "tool":
        client = FakeOpenAI(["policy"], [tool("search_policy", {"query": private_error, "extra": private_error})])
    else:
        client = FakeOpenAI(["policy"], [speech()])
    runtime, store = build_runtime(client, events.append)
    await runtime.start_conversation("demo-failed-turn")
    if stage == "save":
        def fail_save(state):
            raise RuntimeError(private_error)
        store.save = fail_save
    with pytest.raises(error_type):
        await runtime.handle_turn("demo-failed-turn", "What are your clinic hours?")
    assert events[-1].event_type == WorkflowEventType.TURN_FAILED
    assert events[-1].status == "failed" and events[-1].duration_ms >= 0
    assert not any(event.event_type == WorkflowEventType.TURN_COMPLETED for event in events)
    serialized = json.dumps([asdict(event) for event in events])
    assert private_error not in serialized and "secret-key" not in serialized
    if stage in {"router", "model"}:
        assert events[-2].event_type == WorkflowEventType.MODEL_REQUEST_COMPLETED
        assert events[-2].status == "failed"
    if stage == "tool":
        assert events[-2].event_type == WorkflowEventType.TOOL_COMPLETED
        assert events[-2].status == "failed"


@pytest.mark.asyncio
async def test_failed_turn_observer_does_not_replace_original_model_failure():
    original = RuntimeError("PRIVATE_MODEL_EXCEPTION")

    def broken_observer(event):
        raise ValueError("PRIVATE_OBSERVER_EXCEPTION")

    client = FakeOpenAI(["policy"], [original])
    runtime, _ = build_runtime(client, broken_observer)
    await runtime.start_conversation("demo-original-error")
    with pytest.raises(AgentModelRequestError) as caught:
        await runtime.handle_turn("demo-original-error", "Clinic hours?")
    assert caught.value.__cause__ is original


@pytest.mark.asyncio
async def test_successful_tool_execution_status_does_not_claim_a_business_outcome():
    events = []
    client = FakeOpenAI(["policy"], [tool("search_policy", {"query": "prices"}), speech()])
    runtime, _ = build_runtime(client, events.append)
    await runtime.start_conversation("demo-empty-policy")
    await runtime.handle_turn("demo-empty-policy", "What are your prices?")
    result = json.loads(client.responses.create_calls[-1]["input"][-1]["output"])
    assert result == {"found": False, "evidence": []}
    completed, = [event for event in events if event.event_type == WorkflowEventType.TOOL_COMPLETED]
    assert completed.status == "succeeded"


@pytest.mark.asyncio
async def test_concurrent_conversations_do_not_share_hook_scopes():
    events = []
    first, _ = build_runtime(FakeOpenAI(*script(0)), events.append)
    second, _ = build_runtime(FakeOpenAI(["verification"], [tool("lookup_customer", {"phone": "202-555-0142"}), speech()]), events.append)
    await asyncio.gather(first.start_conversation("demo-first"), second.start_conversation("demo-second"))
    await asyncio.gather(first.handle_turn("demo-first", "Hours?"), second.handle_turn("demo-second", "202-555-0142"))
    assert selected_workflow([event for event in events if event.conversation_id == "demo-first"]) == [
        ("demo-first", 1, "policy", None), ("demo-first", 1, "policy", "search_policy"),
    ]
    assert selected_workflow([event for event in events if event.conversation_id == "demo-second"]) == [
        ("demo-second", 1, "verification", None), ("demo-second", 1, "verification", "lookup_customer"),
    ]
    assert all(event.turn_number == 1 for event in events)
    assert all(event.specialist in {None, "policy"} for event in events if event.conversation_id == "demo-first")
    assert all(event.specialist in {None, "verification"} for event in events if event.conversation_id == "demo-second")


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid, error", [
    ("unknown", UnknownAgentToolError), ("malformed", MalformedToolCallError), ("limit", ToolLoopLimitError),
])
async def test_hook_records_only_calls_accepted_for_execution_and_cleans_up_on_errors(invalid, error):
    events = []
    response = tool("lookup_customer" if invalid != "unknown" else "unapproved", {"phone": "202-555-0142"})
    if invalid == "malformed":
        response.output[0].arguments = "[]"
    client = FakeOpenAI([], [response])
    handler = AsyncMock()
    with pytest.raises(error):
        with observe_turn(events.append, "demo-rejected-tool", 1):
            observe_specialist("verification")
            await run_agent_turn(
                client=client, model="model", instructions="Stable rules", user_message="caller",
                tools=[], tool_handlers={"lookup_customer": handler},
                max_tool_rounds=0 if invalid == "limit" else 3,
            )
    assert selected_workflow(events) == [("demo-rejected-tool", 1, "verification", None)]
    assert events[-1].event_type == WorkflowEventType.TURN_FAILED
    before = list(events)
    handler.assert_not_awaited()
    observe_specialist("policy")
    with observe_tool("search_policy"):
        pass
    assert events == before


@pytest.mark.asyncio
async def test_each_case_gets_fresh_state_even_when_runner_is_reused():
    routes, outputs = script(2)
    policy_routes, policy_outputs = script(0)
    client = FakeOpenAI(routes + policy_routes, outputs + policy_outputs)
    run = runner(client)
    routine = await run(load_conversation_cases()[2])
    policy = await run(load_conversation_cases()[0])
    assert routine.booking_confirmed
    assert not policy.verified_customer and not policy.booking_confirmed and not policy.handoff_queued
    assert "Customer verified: false" in client.responses.parse_calls[-1]["instructions"]


@pytest.mark.asyncio
async def test_demo_integrations_return_only_fictional_data_and_preserve_booking_idempotency():
    crm = DemoCRM()
    first = await crm.find_customer_by_phone("202-555-0142")
    second = await crm.find_customer_by_phone("202-555-0143")
    assert first.customer_id.startswith("demo-") and second.customer_id != first.customer_id
    assert first.pets[0].name == "Maize" and first.pets[0].pet_id.startswith("demo-")
    assert await crm.find_customer_by_phone("212-555-9999") is None
    retriever = DemoPolicyRetriever()
    evidence = await retriever.search("clinic hours")
    assert len(evidence) == 1 and evidence[0].chunk.chunk_id == "demo-hours-1"
    assert await retriever.search("clinic prices") == []
    scheduler = DemoScheduler()
    slots = await scheduler.find_slots(first.pets[0].pet_id, AppointmentType.ROUTINE)
    request = BookingRequest(slot_id=slots[0].slot_id, pet_id=first.pets[0].pet_id, confirmed_by_caller=True)
    confirmation = await scheduler.book_appointment(request, "demo-retry-key")
    assert await scheduler.book_appointment(request, "demo-retry-key") == confirmation
    assert confirmation.booking_id.startswith("demo-")
    assert [slot.slot_id for slot in await scheduler.find_slots(first.pets[0].pet_id, AppointmentType.ROUTINE)] == [slots[1].slot_id]
    with pytest.raises(ControlledIntegrationError, match="conflicts"):
        await scheduler.book_appointment(request.model_copy(update={"slot_id": slots[1].slot_id}), "demo-retry-key")
    receipt = await DemoHandoff().create_handoff(HandoffRequest(
        conversation_id="demo-handoff-conversation", reason="caller_request", explanation="The caller requested a person.",
    ))
    assert receipt.conversation_id == "demo-handoff-conversation"
    assert receipt.case_id.startswith("demo-") and receipt.status == "queued"


@pytest.mark.asyncio
async def test_early_completion_does_not_invent_missing_specialist_observations():
    case = load_conversation_cases()[1]
    client = FakeOpenAI(["human_handoff"], [
        tool("request_handoff", {"reason": "caller_request", "explanation": "Scripted explicit request."}), speech(),
    ])
    with pytest.raises(LiveConversationEvaluationError, match="workflow could not complete"):
        await runner(client)(case)
    assert len(client.responses.parse_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["router", "model", "integration"])
async def test_workflow_errors_produce_safe_messages_and_no_exception_chain(stage):
    raw = "SECRET_KEY RAW_PROMPT RAW_ARGUMENTS RAW_RESULTS UPSTREAM_EXCEPTION"
    if stage == "router":
        client = FakeOpenAI([RuntimeError(raw)], [])
    elif stage == "model":
        client = FakeOpenAI(["policy"], [RuntimeError(raw)])
    else:
        client = FakeOpenAI(["policy"], [tool("search_policy", {"query": raw, "extra": raw})])
    with pytest.raises(LiveConversationEvaluationError) as caught:
        await runner(client)(load_conversation_cases()[0])
    assert str(caught.value) == "The controlled conversation workflow could not complete"
    assert caught.value.__suppress_context__ is True
    assert caught.value.__cause__ is None


@pytest.fixture
def configured_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-eval-secret-key")
    monkeypatch.setenv("OPENAI_ROUTER_MODEL", "env-router")
    monkeypatch.setenv("OPENAI_SPECIALIST_MODEL", "env-specialist")


def install_fake_client(monkeypatch, *indices):
    routes, outputs = [], []
    for index in indices:
        case_routes, case_outputs = script(index)
        routes.extend(case_routes)
        outputs.extend(case_outputs)
    client = FakeOpenAI(routes, outputs)
    constructions = []

    def create(**kwargs):
        constructions.append(kwargs)
        return client

    monkeypatch.setattr(live_conversation_eval, "AsyncOpenAI", create)
    return client, constructions


@pytest.mark.parametrize("args, missing_env, error", [
    (["--case-id", "unverified_policy_hours"], None, "--confirm-live"),
    (["--confirm-live", "--all-cases"], "OPENAI_API_KEY", "OPENAI_API_KEY"),
    (["--confirm-live", "--all-cases"], "OPENAI_ROUTER_MODEL", "OPENAI_ROUTER_MODEL"),
    (["--confirm-live", "--all-cases"], "OPENAI_SPECIALIST_MODEL", "OPENAI_SPECIALIST_MODEL"),
    (["--confirm-live"], None, "--case-id"),
    (["--confirm-live", "--all-cases", "--case-id", "unverified_policy_hours"], None, "not allowed"),
    (["--confirm-live", "--case-id", "unknown-case"], None, "Unknown conversation case IDs"),
])
def test_cli_refuses_missing_configuration_and_invalid_selection_before_client_creation(monkeypatch, capsys, configured_env, args, missing_env, error):
    client, constructions = install_fake_client(monkeypatch)
    if missing_env:
        monkeypatch.delenv(missing_env)
    with pytest.raises(SystemExit) as caught:
        live_conversation_eval.main(args)
    assert caught.value.code == 2
    assert error in capsys.readouterr().err
    assert constructions == [] and client.responses.order == []


def test_cli_selected_case_uses_existing_scorer_and_one_shared_client(monkeypatch, capsys, configured_env):
    client, constructions = install_fake_client(monkeypatch, 0)
    reports = []
    original = live_conversation_eval.run_conversation_evaluation

    async def capture(cases, run):
        report = await original(cases, run)
        reports.append(report)
        return report

    monkeypatch.setattr(live_conversation_eval, "run_conversation_evaluation", capture)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://forbidden.example")
    status = live_conversation_eval.main(["--confirm-live", "--case-id", "unverified_policy_hours"])
    assert status == 0
    assert constructions == [{"api_key": "test-eval-secret-key", "base_url": "https://api.openai.com/v1"}]
    assert client.closed
    assert len(client.responses.parse_calls) == 1 and len(client.responses.create_calls) == 2
    assert reports[0]["total"] == reports[0]["passed"] == 1
    output = capsys.readouterr()
    assert "Total: 1" in output.out and "complete_case: 1/1 (100.0%)" in output.out
    assert "test-eval-secret-key" not in output.out + output.err


def test_cli_explicit_models_override_environment(monkeypatch, configured_env):
    client, _ = install_fake_client(monkeypatch, 0)
    assert live_conversation_eval.main([
        "--confirm-live", "--case-id", "unverified_policy_hours",
        "--router-model", " argument-router ", "--specialist-model", " argument-specialist ",
    ]) == 0
    assert client.responses.parse_calls[0]["model"] == "argument-router"
    assert all(request["model"] == "argument-specialist" for request in client.responses.create_calls)


def test_cli_accepts_explicit_models_without_model_environment_variables(monkeypatch, configured_env):
    monkeypatch.delenv("OPENAI_ROUTER_MODEL")
    monkeypatch.delenv("OPENAI_SPECIALIST_MODEL")
    install_fake_client(monkeypatch, 0)
    assert live_conversation_eval.main([
        "--confirm-live", "--case-id", "unverified_policy_hours",
        "--router-model", "argument-router", "--specialist-model", "argument-specialist",
    ]) == 0


@pytest.mark.parametrize("option", ["--router-model", "--specialist-model"])
def test_cli_rejects_blank_model_names_before_network(monkeypatch, configured_env, option):
    _, constructions = install_fake_client(monkeypatch)
    with pytest.raises(SystemExit):
        live_conversation_eval.main(["--confirm-live", "--all-cases", option, " "])
    assert constructions == []


def test_cli_multiple_case_ids_deduplicate_and_run_sequentially_in_dataset_order(monkeypatch, configured_env):
    client, _ = install_fake_client(monkeypatch, 0, 1)
    assert live_conversation_eval.main([
        "--confirm-live", "--case-id", "verified_urgent_intake_handoff",
        "--case-id", "unverified_policy_hours", "--case-id", "unverified_policy_hours",
    ]) == 0
    assert [request["input"] for request in client.responses.parse_calls] == [
        *load_conversation_cases()[0].caller_turns, *load_conversation_cases()[1].caller_turns,
    ]


def test_cli_all_cases_is_explicit_and_uses_complete_tool_loop(monkeypatch, capsys, configured_env):
    client, _ = install_fake_client(monkeypatch, 0, 1, 2)
    assert live_conversation_eval.main(["--confirm-live", "--all-cases"]) == 0
    assert len(client.responses.parse_calls) == 8
    assert len(client.responses.create_calls) == 17
    assert "Passed: 3" in capsys.readouterr().out


def test_cli_scored_failure_returns_one_without_rewriting_model_behavior(monkeypatch, capsys, configured_env):
    client, _ = install_fake_client(monkeypatch)
    client.responses.routes = ["verification"]
    client.responses.outputs = [speech()]
    assert live_conversation_eval.main(["--confirm-live", "--case-id", "unverified_policy_hours"]) == 1
    output = capsys.readouterr()
    assert 'specialist_sequence: expected ["policy"], received ["verification"]' in output.out
    assert 'tool_sequence: expected [["search_policy"]], received [[]]' in output.out


@pytest.mark.parametrize("stage", ["client", "router", "specialist", "controlled_integration"])
def test_cli_safe_failures_never_print_upstream_data(monkeypatch, capsys, configured_env, stage):
    secret = "test-eval-secret-key RAW_PROMPT RAW_ARGUMENTS RAW_RESULTS UPSTREAM_EXCEPTION"
    client, _ = install_fake_client(monkeypatch)
    if stage == "client":
        def fail(**kwargs):
            raise RuntimeError(secret)
        monkeypatch.setattr(live_conversation_eval, "AsyncOpenAI", fail)
    elif stage == "router":
        client.responses.routes = [RuntimeError(secret)]
    elif stage == "specialist":
        client.responses.routes = ["policy"]
        client.responses.outputs = [RuntimeError(secret)]
    else:
        client.responses.routes = ["policy"]
        client.responses.outputs = [tool("search_policy", {"query": "clinic hours"})]

        async def fail_search(*args, **kwargs):
            raise ControlledIntegrationError(secret)

        monkeypatch.setattr(DemoPolicyRetriever, "search", fail_search)
    assert live_conversation_eval.main(["--confirm-live", "--case-id", "unverified_policy_hours"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Live conversation evaluation failed; no report was produced.\n"
