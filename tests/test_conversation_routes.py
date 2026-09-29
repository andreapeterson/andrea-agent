from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_agent_orchestrator
from app.integrations import TurnInterpretationRequestError, TurnInterpretationResponseError
from app.main import app
from app.models.agent_turn import ExtractedIntakeUpdates, TurnUnderstanding
from app.models.appointment import AppointmentSlot, AppointmentType, BookingConfirmation, BookingStatus
from app.models.conversation import ConversationPhase, ConversationState
from app.models.customer import Customer, Pet, PetSpecies
from app.models.policy_answer import PolicyAnswerResponse, PolicyAnswerStatus
from app.services import AgentStateError, AgentToolError
from app.services.agent_orchestrator import AgentOrchestrator
from app.services.conversation_store import InMemoryConversationStore


class SequenceTurnInterpreter:
    def __init__(self, interpretations: list[TurnUnderstanding]) -> None:
        self._interpretations = list(interpretations)
        self.calls: list[str] = []

    async def interpret(self, state: ConversationState, user_message: str) -> TurnUnderstanding:
        self.calls.append(user_message)
        if not self._interpretations:
            return TurnUnderstanding(
                intents=[],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(),
                policy_question=None,
                appointment_selection=None,
                booking_confirmed=None,
            )
        return self._interpretations.pop(0)


class FakeLegacyCRMClient:
    def __init__(self, customer: Customer | None) -> None:
        self.customer = customer

    async def find_customer_by_phone(self, phone: str) -> Customer | None:
        return self.customer


class FakeSchedulerClient:
    def __init__(self, *, slots: list[AppointmentSlot] | None = None) -> None:
        self.slots = slots or []
        self.slot_calls: list[tuple[str, AppointmentType]] = []

    async def find_slots(self, pet_id: str, appointment_type: AppointmentType) -> list[AppointmentSlot]:
        self.slot_calls.append((pet_id, appointment_type))
        return list(self.slots)

    async def book_appointment(self, booking_request: object, idempotency_key: str) -> BookingConfirmation:
        return BookingConfirmation(
            booking_id="booking-1",
            slot_id=booking_request.slot_id,
            pet_id=booking_request.pet_id,
            status=BookingStatus.CONFIRMED,
        )


class FakeHandoffClient:
    async def create_handoff(self, handoff_request: object) -> object:
        return handoff_request


class FakePolicyService:
    def __init__(self, response: PolicyAnswerResponse | None = None) -> None:
        self.response = response or PolicyAnswerResponse(
            question="Can I cancel my appointment?",
            answer="The clinic allows cancellations with notice.",
            status=PolicyAnswerStatus.ANSWERED,
            citations=[],
        )
        self.calls: list[str] = []

    async def answer(self, question: str) -> PolicyAnswerResponse:
        self.calls.append(question)
        return self.response


class FailingOrchestrator:
    async def start_conversation(self, conversation_id: str):
        raise AgentStateError("invalid state")

    async def handle_turn(self, conversation_id: str, user_message: str):
        raise TurnInterpretationRequestError("upstream unavailable")


class MalformedTurnInterpreter:
    async def interpret(self, state: ConversationState, user_message: str):
        raise TurnInterpretationResponseError("invalid structured output")


def make_customer() -> Customer:
    return Customer(
        customer_id="customer_1001",
        first_name="Andrea",
        last_name="Peterson",
        phone_number="3212222222",
        pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)],
    )


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_conversation_start_route_returns_public_response(client: TestClient) -> None:
    orchestrator = AgentOrchestrator(
        conversation_store=InMemoryConversationStore(),
        turn_interpreter=SequenceTurnInterpreter([]),
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "idemp-1",
    )
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator

    response = client.post("/conversations")

    assert response.status_code == 201
    payload = response.json()
    assert payload["phase"] == ConversationPhase.VERIFYING_CUSTOMER.value
    assert payload["assistant_message"] == "Please provide the caller's phone number."
    assert payload["conversation_id"].startswith("conversation-")
    assert payload["turn_count"] == 0


def test_conversation_messages_work_across_turns(client: TestClient) -> None:
    customer = make_customer()
    orchestrator = AgentOrchestrator(
        conversation_store=InMemoryConversationStore(),
        turn_interpreter=SequenceTurnInterpreter(
            [
                TurnUnderstanding(
                    intents=[],
                    phone_number="3212222222",
                    pet_reference=None,
                    original_concern=None,
                    intake_updates=ExtractedIntakeUpdates(),
                    policy_question=None,
                    appointment_selection=None,
                    booking_confirmed=None,
                ),
                TurnUnderstanding(
                    intents=[],
                    phone_number=None,
                    pet_reference=None,
                    original_concern="Baxter is weak and not eating.",
                    intake_updates=ExtractedIntakeUpdates(),
                    policy_question=None,
                    appointment_selection=None,
                    booking_confirmed=None,
                ),
            ]
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "idemp-1",
    )
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator

    start = client.post("/conversations")
    conversation_id = start.json()["conversation_id"]

    first = client.post(f"/conversations/{conversation_id}/messages", json={"message": "My number is 321-222-2222"})
    second = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Baxter is weak and not eating."})

    assert first.status_code == 200
    assert first.json()["phase"] == ConversationPhase.COLLECTING_CONCERN.value
    assert second.status_code == 200
    assert second.json()["phase"] == ConversationPhase.COLLECTING_INTAKE.value
    assert second.json()["assistant_message"].startswith("I have the concern")


def test_conversation_messages_reject_blank_input(client: TestClient) -> None:
    orchestrator = AgentOrchestrator(
        conversation_store=InMemoryConversationStore(),
        turn_interpreter=SequenceTurnInterpreter([]),
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "idemp-1",
    )
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator

    response = client.post("/conversations/conv-1/messages", json={"message": "   "})

    assert response.status_code == 422


def test_unknown_conversation_returns_404(client: TestClient) -> None:
    orchestrator = AgentOrchestrator(
        conversation_store=InMemoryConversationStore(),
        turn_interpreter=SequenceTurnInterpreter([]),
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "idemp-1",
    )
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator

    response = client.post("/conversations/missing/messages", json={"message": "Hello there"})

    assert response.status_code == 404
    assert response.json()["detail"] == "conversation-not-found"


def test_policy_answer_is_returned_without_leaking_internal_state(client: TestClient) -> None:
    customer = make_customer()
    orchestrator = AgentOrchestrator(
        conversation_store=InMemoryConversationStore(),
        turn_interpreter=SequenceTurnInterpreter(
            [
                TurnUnderstanding(
                    intents=[],
                    phone_number=None,
                    pet_reference=None,
                    original_concern=None,
                    intake_updates=ExtractedIntakeUpdates(),
                    policy_question="Can I cancel my appointment?",
                    appointment_selection=None,
                    booking_confirmed=None,
                )
            ]
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(
            response=PolicyAnswerResponse(
                question="Can I cancel my appointment?",
                answer="The clinic allows cancellations with notice.",
                status=PolicyAnswerStatus.ANSWERED,
                citations=[],
            )
        ),
        idempotency_key_factory=lambda: "idemp-1",
    )
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator

    start = client.post("/conversations")
    conversation_id = start.json()["conversation_id"]
    response = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Can I cancel my appointment?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["policy_answer"]["answer"] == "The clinic allows cancellations with notice."
    assert payload["phase"] == ConversationPhase.VERIFYING_CUSTOMER.value
    assert "customer_id" not in payload
    assert "verified_customer" not in payload


def test_upstream_agent_failures_are_mapped_to_http_results(client: TestClient) -> None:
    app.dependency_overrides[get_agent_orchestrator] = lambda: FailingOrchestrator()

    response = client.post("/conversations/conv-1/messages", json={"message": "hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == "agent-service-unavailable"


def test_invalid_turn_understanding_response_maps_to_502(client: TestClient) -> None:
    orchestrator = AgentOrchestrator(
        conversation_store=InMemoryConversationStore(),
        turn_interpreter=MalformedTurnInterpreter(),
        legacy_crm_client=FakeLegacyCRMClient(make_customer()),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "idemp-1",
    )
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator

    start = client.post("/conversations")
    conversation_id = start.json()["conversation_id"]
    response = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "invalid-agent-understanding"
