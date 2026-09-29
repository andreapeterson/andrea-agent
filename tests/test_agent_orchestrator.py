import asyncio
from datetime import datetime

import pytest

from app.models.agent_turn import ExtractedIntakeUpdates, TurnIntent, TurnUnderstanding
from app.models.appointment import AppointmentSlot, AppointmentType, BookingConfirmation, BookingRequest, BookingStatus
from app.models.conversation import ConversationPhase, ConversationState
from app.models.customer import Customer, Pet, PetSpecies
from app.models.handoff import HandoffReceipt, HandoffStatus
from app.models.policy_answer import PolicyAnswerResponse, PolicyAnswerStatus
from app.models.routing import IntakeAnswers
from app.services.agent_orchestrator import AgentOrchestrator, AgentStateError
from app.services.conversation_store import ConversationNotFoundError, InMemoryConversationStore


class FakeTurnInterpreter:
    def __init__(self, interpretation: TurnUnderstanding | None = None, *, delay_event: asyncio.Event | None = None) -> None:
        self._interpretation = interpretation
        self.delay_event = delay_event
        self.calls: list[str] = []

    async def interpret(self, state: ConversationState, user_message: str) -> TurnUnderstanding:
        self.calls.append(user_message)
        if self.delay_event is not None:
            await self.delay_event.wait()
        if self._interpretation is not None:
            return self._interpretation
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


class FakeLegacyCRMClient:
    def __init__(self, customer: Customer | None) -> None:
        self.customer = customer
        self.calls: list[str] = []

    async def find_customer_by_phone(self, phone: str) -> Customer | None:
        self.calls.append(phone)
        return self.customer


class FakeSchedulerClient:
    def __init__(self, *, slots: list[AppointmentSlot] | None = None, booking_confirmation: BookingConfirmation | None = None) -> None:
        self.slots = slots or []
        self.booking_confirmation = booking_confirmation
        self.slot_calls: list[tuple[str, AppointmentType]] = []
        self.book_calls: list[BookingRequest] = []

    async def find_slots(self, pet_id: str, appointment_type: AppointmentType) -> list[AppointmentSlot]:
        self.slot_calls.append((pet_id, appointment_type))
        return list(self.slots)

    async def book_appointment(self, booking_request: BookingRequest, idempotency_key: str) -> BookingConfirmation:
        self.book_calls.append(booking_request)
        if self.booking_confirmation is not None:
            return self.booking_confirmation
        return BookingConfirmation(
            booking_id="booking-1",
            slot_id=booking_request.slot_id,
            pet_id=booking_request.pet_id,
            status=BookingStatus.CONFIRMED,
        )


class FakeHandoffClient:
    def __init__(self, receipt: HandoffReceipt | None = None) -> None:
        self.receipt = receipt
        self.requests: list[object] = []

    async def create_handoff(self, handoff_request: object) -> HandoffReceipt:
        self.requests.append(handoff_request)
        if self.receipt is not None:
            return self.receipt
        return HandoffReceipt(
            case_id="case-1",
            conversation_id="conv-1",
            status=HandoffStatus.QUEUED,
            received_at=datetime(2026, 9, 27, 12, 0, 0),
        )


class FakePolicyService:
    def __init__(self, response: PolicyAnswerResponse | None = None) -> None:
        self.response = response or PolicyAnswerResponse(
            question="What is the cancellation policy?",
            answer="The clinic allows cancellations with notice.",
            status=PolicyAnswerStatus.ANSWERED,
            citations=[],
        )
        self.calls: list[str] = []

    async def answer(self, question: str) -> PolicyAnswerResponse:
        self.calls.append(question)
        return self.response


def make_customer(*, pets: list[Pet]) -> Customer:
    return Customer(
        customer_id="customer_1001",
        first_name="Andrea",
        last_name="Peterson",
        phone_number="3212222222",
        pets=pets,
    )


@pytest.mark.asyncio
async def test_blank_user_message_is_rejected_without_changing_store() -> None:
    store = InMemoryConversationStore()
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(),
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    with pytest.raises(AgentStateError):
        await orchestrator.handle_turn("conv-blank", "   ")

    with pytest.raises(ConversationNotFoundError):
        store.get("conv-blank")


@pytest.mark.asyncio
async def test_terminal_phase_checks_happen_before_the_interpreter_runs() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-complete")
    state.phase = ConversationPhase.COMPLETED
    store.save(state)
    interpreter = FakeTurnInterpreter()
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=interpreter,
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    result = await orchestrator.handle_turn("conv-complete", "This should be ignored.")

    assert interpreter.calls == []
    assert result.state.phase == ConversationPhase.COMPLETED
    assert result.assistant_message == "This conversation is already complete."


@pytest.mark.asyncio
async def test_phone_number_triggers_customer_lookup_and_single_pet_autoselect() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    crm = FakeLegacyCRMClient(customer)
    orchestrator = AgentOrchestrator(
        conversation_store=InMemoryConversationStore(),
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.PROVIDE_PHONE],
                phone_number="321-222-2222",
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(),
                policy_question=None,
                appointment_selection=None,
                booking_confirmed=None,
            )
        ),
        legacy_crm_client=crm,
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    await orchestrator.start_conversation("conv-1")
    result = await orchestrator.handle_turn("conv-1", "My number is 321-222-2222")
    assert crm.calls == ["321-222-2222"]
    assert result.state.verified_customer is not None
    assert result.state.selected_pet_id == "pet_2001"
    assert result.state.phase == ConversationPhase.COLLECTING_CONCERN


@pytest.mark.asyncio
async def test_urgent_routing_creates_handoff_without_scheduler() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create("conv-2")
    state.phase = ConversationPhase.COLLECTING_INTAKE
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    state.original_concern = "He cannot breathe normally."
    state.intake_answers = IntakeAnswers(difficulty_breathing=True)
    store.save(state)
    scheduler = FakeSchedulerClient(slots=[AppointmentSlot(
        slot_id="slot-1",
        clinic_id="clinic-1",
        starts_at=datetime(2026, 9, 27, 9, 0, 0),
        ends_at=datetime(2026, 9, 27, 9, 30, 0),
        appointment_type=AppointmentType.SAME_DAY,
    )])
    handoff_client = FakeHandoffClient()
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.ANSWER_INTAKE],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(difficulty_breathing=True),
                policy_question=None,
                appointment_selection=None,
                booking_confirmed=None,
            )
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=scheduler,
        handoff_client=handoff_client,
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    result = await orchestrator.handle_turn("conv-2", "He is having trouble breathing.")
    assert result.state.phase == ConversationPhase.HANDOFF_COMPLETE
    assert result.state.handoff_receipt is not None
    assert len(scheduler.slot_calls) == 0
    assert len(handoff_client.requests) == 1


@pytest.mark.asyncio
async def test_explicit_confirmation_books_with_idempotency_key_saved_before_scheduler_call() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create("conv-3")
    state.phase = ConversationPhase.CONFIRMING_BOOKING
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    state.offered_slots = [
        AppointmentSlot(
            slot_id="slot-1",
            clinic_id="clinic-1",
            starts_at=datetime(2026, 9, 27, 9, 0, 0),
            ends_at=datetime(2026, 9, 27, 9, 30, 0),
            appointment_type=AppointmentType.SAME_DAY,
        )
    ]
    state.selected_slot_id = "slot-1"
    store.save(state)

    class BookingScheduler(FakeSchedulerClient):
        async def book_appointment(self, booking_request: BookingRequest, idempotency_key: str) -> BookingConfirmation:
            persisted = store.get("conv-3")
            assert persisted.booking_idempotency_key == "key-book-1"
            return await super().book_appointment(booking_request, idempotency_key)

    scheduler = BookingScheduler(
        booking_confirmation=BookingConfirmation(
            booking_id="booking-1",
            slot_id="slot-1",
            pet_id="pet_2001",
            status=BookingStatus.CONFIRMED,
        )
    )
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.CONFIRM_BOOKING],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(),
                policy_question=None,
                appointment_selection=None,
                booking_confirmed=True,
            )
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=scheduler,
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-book-1",
    )

    result = await orchestrator.handle_turn("conv-3", "Yes, please book it.")
    assert result.state.phase == ConversationPhase.BOOKING_COMPLETE
    assert result.state.booking_confirmation is not None
    assert result.state.booking_idempotency_key == "key-book-1"


@pytest.mark.asyncio
async def test_same_conversation_turns_are_serialized_by_lock() -> None:
    store = InMemoryConversationStore()
    event = asyncio.Event()

    class BlockingInterpreter(FakeTurnInterpreter):
        async def interpret(self, state: ConversationState, user_message: str):
            await event.wait()
            return await super().interpret(state, user_message)

    interpreter = BlockingInterpreter()
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=interpreter,
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    await orchestrator.start_conversation("conv-lock")
    first = asyncio.create_task(orchestrator.handle_turn("conv-lock", "first message"))
    await asyncio.sleep(0)
    second = asyncio.create_task(orchestrator.handle_turn("conv-lock", "second message"))
    assert not second.done()
    event.set()
    await asyncio.gather(first, second)
    assert store.get("conv-lock").turn_count == 2
