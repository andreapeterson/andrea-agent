from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_agent_orchestrator
from app.integrations import TurnInterpretationRequestError, TurnInterpretationResponseError
from app.main import app
from app.models.agent_turn import ExtractedIntakeUpdates, TurnUnderstanding
from app.models.appointment import (
    AppointmentSlot,
    AppointmentType,
    BookingConfirmation,
    BookingRequest,
    BookingStatus,
)
from app.models.conversation import ConversationPhase, ConversationState
from app.models.customer import Customer, Pet, PetSpecies
from app.models.handoff import HandoffReceipt, HandoffStatus
from app.models.policy_answer import PolicyAnswerResponse, PolicyAnswerStatus
from app.services import AgentStateError, AgentToolError, ConversationNotFoundError
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
        self.calls: list[str] = []

    async def find_customer_by_phone(self, phone: str) -> Customer | None:
        self.calls.append(phone)
        return self.customer


class FakeSchedulerClient:
    def __init__(
        self,
        *,
        slots: list[AppointmentSlot] | None = None,
        store: InMemoryConversationStore | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.slots = slots or []
        self.slot_calls: list[tuple[str, AppointmentType]] = []
        self.book_calls: list[BookingRequest] = []
        self.idempotency_keys: list[str] = []
        self.store = store
        self.conversation_id = conversation_id

    async def find_slots(self, pet_id: str, appointment_type: AppointmentType) -> list[AppointmentSlot]:
        self.slot_calls.append((pet_id, appointment_type))
        return list(self.slots)

    async def book_appointment(self, booking_request: BookingRequest, idempotency_key: str) -> BookingConfirmation:
        if self.store is not None and self.conversation_id is not None:
            assert self.store.get(self.conversation_id).booking_idempotency_key == idempotency_key
        self.book_calls.append(booking_request)
        self.idempotency_keys.append(idempotency_key)
        return BookingConfirmation(
            booking_id="booking-1",
            slot_id=booking_request.slot_id,
            pet_id=booking_request.pet_id,
            status=BookingStatus.CONFIRMED,
        )


class FakeHandoffClient:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def create_handoff(self, handoff_request: object) -> HandoffReceipt:
        self.requests.append(handoff_request)
        return HandoffReceipt(
            case_id="case-1",
            conversation_id=handoff_request.conversation_id,
            status=HandoffStatus.QUEUED,
            received_at=datetime(2026, 9, 28, 12, 0),
        )


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


class RaisingOrchestrator:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def handle_turn(self, conversation_id: str, user_message: str):
        raise self.error


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


def understanding(
    *,
    phone_number: str | None = None,
    original_concern: str | None = None,
    intake_updates: ExtractedIntakeUpdates | None = None,
    policy_question: str | None = None,
    appointment_selection: str | None = None,
    booking_confirmed: bool | None = None,
) -> TurnUnderstanding:
    return TurnUnderstanding(
        intents=[],
        phone_number=phone_number,
        pet_reference=None,
        original_concern=original_concern,
        intake_updates=intake_updates or ExtractedIntakeUpdates(),
        policy_question=policy_question,
        appointment_selection=appointment_selection,
        booking_confirmed=booking_confirmed,
    )


def routine_slot() -> AppointmentSlot:
    return AppointmentSlot(
        slot_id="slot-routine-1",
        clinic_id="clinic-1",
        starts_at=datetime(2026, 9, 29, 10, 0),
        ends_at=datetime(2026, 9, 29, 10, 45),
        appointment_type=AppointmentType.ROUTINE,
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


@pytest.mark.parametrize(
    ("error_type", "status_code", "detail"),
    [
        (TurnInterpretationRequestError, 503, "agent-understanding-unavailable"),
        (TurnInterpretationResponseError, 502, "invalid-agent-understanding"),
        (AgentToolError, 503, "agent-tool-unavailable"),
        (AgentStateError, 409, "invalid-conversation-state"),
        (ConversationNotFoundError, 404, "conversation-not-found"),
    ],
)
def test_conversation_errors_map_to_specific_sanitized_details(
    client: TestClient,
    error_type: type[Exception],
    status_code: int,
    detail: str,
) -> None:
    app.dependency_overrides[get_agent_orchestrator] = lambda: RaisingOrchestrator(error_type("sensitive upstream text"))

    response = client.post("/conversations/conv-1/messages", json={"message": "hello"})

    assert response.status_code == status_code
    assert response.json() == {"detail": detail}


def test_missing_agent_orchestrator_maps_to_503(client: TestClient) -> None:
    app.dependency_overrides[get_agent_orchestrator] = lambda: None

    response = client.post("/conversations/conv-1/messages", json={"message": "hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "agent-service-unavailable"}


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


def test_complete_routine_booking_journey_and_terminal_guard(client: TestClient) -> None:
    store = InMemoryConversationStore()
    customer = make_customer()
    crm = FakeLegacyCRMClient(customer)
    policy = FakePolicyService()
    handoff = FakeHandoffClient()
    slot = routine_slot()
    interpreter = SequenceTurnInterpreter(
        [
            understanding(phone_number="3212222222"),
            understanding(original_concern="Baxter is due for a routine visit."),
            understanding(
                intake_updates=ExtractedIntakeUpdates(
                    difficulty_breathing=False,
                    uncontrolled_bleeding=False,
                    collapsed_or_unresponsive=False,
                    known_toxin_exposure=False,
                    rapidly_worsening=False,
                )
            ),
            understanding(appointment_selection=slot.slot_id),
            understanding(booking_confirmed=True),
        ]
    )
    scheduler = FakeSchedulerClient(slots=[slot], store=store)
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=interpreter,
        legacy_crm_client=crm,
        scheduler_client=scheduler,
        handoff_client=handoff,
        policy_service=policy,
        idempotency_key_factory=lambda: "routine-booking-key",
    )
    scheduler.conversation_id = "pending"
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator

    start = client.post("/conversations")
    conversation_id = start.json()["conversation_id"]
    scheduler.conversation_id = conversation_id

    verified = client.post(f"/conversations/{conversation_id}/messages", json={"message": "My number is 3212222222"})
    concern = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Baxter needs a routine visit."})
    intake = client.post(f"/conversations/{conversation_id}/messages", json={"message": "No to all five urgent questions."})
    selected = client.post(f"/conversations/{conversation_id}/messages", json={"message": "The first slot, please."})
    booked = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Yes, book that appointment."})

    assert verified.status_code == concern.status_code == intake.status_code == selected.status_code == booked.status_code == 200
    assert intake.json()["phase"] == ConversationPhase.SELECTING_APPOINTMENT.value
    assert scheduler.slot_calls == [("pet_2001", AppointmentType.ROUTINE)]
    assert intake.json()["available_slots"][0]["appointment_type"] == AppointmentType.ROUTINE.value
    assert selected.json()["phase"] == ConversationPhase.CONFIRMING_BOOKING.value
    assert booked.json()["phase"] == ConversationPhase.BOOKING_COMPLETE.value
    assert len(scheduler.book_calls) == 1
    assert scheduler.idempotency_keys == ["routine-booking-key"]
    assert store.get(conversation_id).booking_idempotency_key == "routine-booking-key"

    interpreter_calls = list(interpreter.calls)
    follow_up = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Please book another appointment."})

    assert follow_up.status_code == 200
    assert follow_up.json()["phase"] == ConversationPhase.BOOKING_COMPLETE.value
    assert len(scheduler.book_calls) == 1
    assert crm.calls == ["3212222222"]
    assert policy.calls == []
    assert interpreter.calls == interpreter_calls


def test_complete_urgent_handoff_journey_and_terminal_guard(client: TestClient) -> None:
    customer = make_customer()
    crm = FakeLegacyCRMClient(customer)
    scheduler = FakeSchedulerClient(slots=[routine_slot()])
    handoff = FakeHandoffClient()
    policy = FakePolicyService()
    interpreter = SequenceTurnInterpreter(
        [
            understanding(phone_number="3212222222"),
            understanding(original_concern="Baxter is having trouble breathing."),
            understanding(intake_updates=ExtractedIntakeUpdates(difficulty_breathing=True)),
        ]
    )
    orchestrator = AgentOrchestrator(
        conversation_store=InMemoryConversationStore(),
        turn_interpreter=interpreter,
        legacy_crm_client=crm,
        scheduler_client=scheduler,
        handoff_client=handoff,
        policy_service=policy,
        idempotency_key_factory=lambda: "unused-key",
    )
    app.dependency_overrides[get_agent_orchestrator] = lambda: orchestrator

    start = client.post("/conversations")
    conversation_id = start.json()["conversation_id"]
    verified = client.post(f"/conversations/{conversation_id}/messages", json={"message": "My number is 3212222222"})
    concern = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Baxter is having trouble breathing."})
    urgent = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Yes, he is having trouble breathing."})

    assert verified.status_code == concern.status_code == urgent.status_code == 200
    assert urgent.json()["phase"] == ConversationPhase.HANDOFF_COMPLETE.value
    assert len(handoff.requests) == 1
    assert scheduler.slot_calls == []
    assert scheduler.book_calls == []

    interpreter_calls = list(interpreter.calls)
    follow_up = client.post(f"/conversations/{conversation_id}/messages", json={"message": "Any appointment slots?"})

    assert follow_up.status_code == 200
    assert follow_up.json()["phase"] == ConversationPhase.HANDOFF_COMPLETE.value
    assert len(handoff.requests) == 1
    assert scheduler.slot_calls == []
    assert scheduler.book_calls == []
    assert crm.calls == ["3212222222"]
    assert policy.calls == []
    assert interpreter.calls == interpreter_calls
