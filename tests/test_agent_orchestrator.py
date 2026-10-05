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
from app.integrations.scheduler_client import SchedulerRequestError, SlotUnavailableError
from app.services.agent_orchestrator import AgentOrchestrator, AgentStateError, AgentToolError
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
async def test_policy_question_during_concern_collection_preserves_phase_and_stored_data() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create("conv-policy-concern")
    state.phase = ConversationPhase.COLLECTING_CONCERN
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    store.save(state)
    policy = FakePolicyService()
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.ASK_POLICY],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(),
                policy_question="What is the cancellation policy?",
                appointment_selection=None,
                booking_confirmed=None,
            )
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=policy,
        idempotency_key_factory=lambda: "key-1",
    )

    result = await orchestrator.handle_turn("conv-policy-concern", "Can I cancel my appointment?")

    assert result.state.phase == ConversationPhase.COLLECTING_CONCERN
    assert result.state.verified_customer == customer
    assert result.state.selected_pet_id == "pet_2001"
    assert result.state.original_concern is None
    assert result.assistant_message == (
        "The clinic allows cancellations with notice. Please tell me the pet's concern."
    )
    assert policy.calls == ["What is the cancellation policy?"]


@pytest.mark.parametrize(
    ("phase", "expected_prompt"),
    [
        (ConversationPhase.SELECTING_PET, "Which pet are you calling about? Baxter"),
        (ConversationPhase.COLLECTING_INTAKE, "Is there uncontrolled bleeding?"),
    ],
)
@pytest.mark.asyncio
async def test_policy_question_preserves_pet_selection_and_intake_phases(
    phase: ConversationPhase,
    expected_prompt: str,
) -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create(f"conv-policy-{phase.value}")
    state.phase = phase
    state.verified_customer = customer
    if phase == ConversationPhase.COLLECTING_INTAKE:
        state.selected_pet_id = "pet_2001"
        state.original_concern = "Baxter is unwell."
        state.intake_answers.difficulty_breathing = False
    store.save(state)
    policy = FakePolicyService()
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.ASK_POLICY],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(),
                policy_question="What is the cancellation policy?",
                appointment_selection=None,
                booking_confirmed=None,
            )
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=policy,
        idempotency_key_factory=lambda: "key-1",
    )

    result = await orchestrator.handle_turn(state.conversation_id, "Can I cancel my appointment?")

    assert result.state.phase == phase
    assert expected_prompt in result.assistant_message
    assert policy.calls == ["What is the cancellation policy?"]
    if phase == ConversationPhase.COLLECTING_INTAKE:
        assert result.state.intake_answers.difficulty_breathing is False
        assert result.state.routing_decision is not None
        assert result.state.routing_decision.missing_fields[0] == "uncontrolled_bleeding"


@pytest.mark.asyncio
async def test_policy_question_during_selecting_appointment_preserves_slots_and_state() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create("conv-policy-slots")
    state.phase = ConversationPhase.SELECTING_APPOINTMENT
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    state.original_concern = "Baxter is weak."
    state.intake_answers = IntakeAnswers(
        difficulty_breathing=False,
        uncontrolled_bleeding=False,
        collapsed_or_unresponsive=False,
        known_toxin_exposure=False,
        rapidly_worsening=False,
    )
    state.offered_slots = [
        AppointmentSlot(
            slot_id="slot-1",
            clinic_id="clinic-1",
            starts_at=datetime(2026, 9, 28, 9, 0),
            ends_at=datetime(2026, 9, 28, 9, 30),
            appointment_type=AppointmentType.ROUTINE,
        )
    ]
    state.selected_slot_id = "slot-1"
    state.booking_idempotency_key = "persisted-key"
    store.save(state)
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.ASK_POLICY],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(),
                policy_question="What is the cancellation policy?",
                appointment_selection=None,
                booking_confirmed=None,
            )
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "new-key",
    )

    result = await orchestrator.handle_turn("conv-policy-slots", "Can I cancel my appointment?")

    assert result.state.phase == ConversationPhase.SELECTING_APPOINTMENT
    assert result.state.verified_customer == customer
    assert result.state.selected_pet_id == "pet_2001"
    assert result.state.original_concern == "Baxter is weak."
    assert result.state.intake_answers == state.intake_answers
    assert result.state.offered_slots == state.offered_slots
    assert result.state.selected_slot_id == "slot-1"
    assert result.state.booking_idempotency_key == "persisted-key"
    assert "slot-1" in result.assistant_message


@pytest.mark.asyncio
async def test_policy_question_during_booking_confirmation_does_not_book() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create("conv-policy-confirm")
    state.phase = ConversationPhase.CONFIRMING_BOOKING
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    state.selected_slot_id = "slot-1"
    store.save(state)
    scheduler = FakeSchedulerClient()
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.ASK_POLICY, TurnIntent.CONFIRM_BOOKING],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(),
                policy_question="What is the cancellation policy?",
                appointment_selection=None,
                booking_confirmed=True,
            )
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=scheduler,
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    result = await orchestrator.handle_turn("conv-policy-confirm", "Can I cancel my appointment?")

    assert result.state.phase == ConversationPhase.CONFIRMING_BOOKING
    assert result.state.selected_slot_id == "slot-1"
    assert scheduler.book_calls == []


@pytest.mark.asyncio
async def test_urgent_intake_answer_wins_over_policy_question() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create("conv-policy-urgent")
    state.phase = ConversationPhase.COLLECTING_INTAKE
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    state.original_concern = "Baxter is having trouble breathing."
    store.save(state)
    policy = FakePolicyService()
    scheduler = FakeSchedulerClient()
    handoff = FakeHandoffClient()
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.ANSWER_INTAKE, TurnIntent.ASK_POLICY],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(difficulty_breathing=True),
                policy_question="What is the cancellation policy?",
                appointment_selection=None,
                booking_confirmed=None,
            )
        ),
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=scheduler,
        handoff_client=handoff,
        policy_service=policy,
        idempotency_key_factory=lambda: "key-1",
    )

    result = await orchestrator.handle_turn("conv-policy-urgent", "He is having trouble breathing. Also, can I cancel?")

    assert result.state.phase == ConversationPhase.HANDOFF_COMPLETE
    assert result.state.intake_answers.difficulty_breathing is True
    assert len(handoff.requests) == 1
    assert policy.calls == []
    assert scheduler.slot_calls == []


@pytest.mark.asyncio
async def test_extracted_none_does_not_erase_stored_false_intake_answer() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-intake-merge")
    state.phase = ConversationPhase.COLLECTING_INTAKE
    state.intake_answers.difficulty_breathing = False
    store.save(state)
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=FakeTurnInterpreter(
            TurnUnderstanding(
                intents=[TurnIntent.ANSWER_INTAKE],
                phone_number=None,
                pet_reference=None,
                original_concern=None,
                intake_updates=ExtractedIntakeUpdates(difficulty_breathing=None),
                policy_question=None,
                appointment_selection=None,
                booking_confirmed=None,
            )
        ),
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    result = await orchestrator.handle_turn("conv-intake-merge", "I am not sure about the rest.")

    assert result.state.intake_answers.difficulty_breathing is False


@pytest.mark.asyncio
async def test_booking_confirmation_outside_confirmation_phase_does_not_book() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-wrong-phase-booking")
    state.phase = ConversationPhase.COLLECTING_CONCERN
    store.save(state)
    scheduler = FakeSchedulerClient()
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
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=scheduler,
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    result = await orchestrator.handle_turn("conv-wrong-phase-booking", "Yes, book it.")

    assert result.state.phase == ConversationPhase.COLLECTING_CONCERN
    assert scheduler.book_calls == []


@pytest.mark.asyncio
async def test_booking_retry_reuses_persisted_idempotency_key_after_uncertain_failure() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create("conv-booking-retry")
    state.phase = ConversationPhase.CONFIRMING_BOOKING
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    state.selected_slot_id = "slot-1"
    store.save(state)

    class FlakyScheduler(FakeSchedulerClient):
        def __init__(self) -> None:
            super().__init__()
            self.idempotency_keys: list[str] = []

        async def book_appointment(self, booking_request: BookingRequest, idempotency_key: str) -> BookingConfirmation:
            assert store.get("conv-booking-retry").booking_idempotency_key == idempotency_key
            self.idempotency_keys.append(idempotency_key)
            if len(self.idempotency_keys) == 1:
                raise SchedulerRequestError("uncertain scheduler failure")
            return await super().book_appointment(booking_request, idempotency_key)

    scheduler = FlakyScheduler()
    key_calls: list[str] = []

    def key_factory() -> str:
        generated = f"key-{len(key_calls) + 1}"
        key_calls.append(generated)
        return generated

    interpreter = FakeTurnInterpreter(
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
    )
    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=interpreter,
        legacy_crm_client=FakeLegacyCRMClient(customer),
        scheduler_client=scheduler,
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=key_factory,
    )

    with pytest.raises(AgentToolError):
        await orchestrator.handle_turn("conv-booking-retry", "Yes, book it.")

    failed_state = store.get("conv-booking-retry")
    assert failed_state.phase == ConversationPhase.CONFIRMING_BOOKING
    assert failed_state.booking_idempotency_key == "key-1"

    result = await orchestrator.handle_turn("conv-booking-retry", "Yes, book it.")

    assert result.state.phase == ConversationPhase.BOOKING_COMPLETE
    assert scheduler.idempotency_keys == ["key-1", "key-1"]
    assert key_calls == ["key-1"]


@pytest.mark.asyncio
async def test_disappeared_slot_refreshes_options_without_auto_selecting() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    old_slot = AppointmentSlot(
        slot_id="slot-old",
        clinic_id="clinic-1",
        starts_at=datetime(2026, 9, 28, 9, 0),
        ends_at=datetime(2026, 9, 28, 9, 30),
        appointment_type=AppointmentType.SAME_DAY,
    )
    fresh_slot = AppointmentSlot(
        slot_id="slot-fresh",
        clinic_id="clinic-1",
        starts_at=datetime(2026, 9, 28, 10, 0),
        ends_at=datetime(2026, 9, 28, 10, 30),
        appointment_type=AppointmentType.SAME_DAY,
    )
    store = InMemoryConversationStore()
    state = store.create("conv-disappeared-slot")
    state.phase = ConversationPhase.CONFIRMING_BOOKING
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    state.appointment_type = AppointmentType.SAME_DAY
    state.offered_slots = [old_slot]
    state.selected_slot_id = old_slot.slot_id
    state.booking_idempotency_key = "old-idempotency-key"
    store.save(state)

    class DisappearingScheduler(FakeSchedulerClient):
        def __init__(self) -> None:
            super().__init__(slots=[fresh_slot])
            self.booked_slot_ids: list[str] = []

        async def book_appointment(self, booking_request: BookingRequest, idempotency_key: str) -> BookingConfirmation:
            self.booked_slot_ids.append(booking_request.slot_id)
            raise SlotUnavailableError("slot unavailable")

    scheduler = DisappearingScheduler()
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
        idempotency_key_factory=lambda: "new-idempotency-key",
    )

    result = await orchestrator.handle_turn("conv-disappeared-slot", "Yes, book it.")

    assert scheduler.booked_slot_ids == ["slot-old"]
    assert scheduler.slot_calls == [("pet_2001", AppointmentType.SAME_DAY)]
    assert result.state.phase == ConversationPhase.SELECTING_APPOINTMENT
    assert result.state.selected_slot_id is None
    assert result.state.booking_idempotency_key is None
    assert result.state.offered_slots == [fresh_slot]
    assert "slot-fresh" in result.assistant_message


@pytest.mark.asyncio
async def test_slot_refresh_failure_is_wrapped_as_agent_tool_error() -> None:
    customer = make_customer(pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)])
    store = InMemoryConversationStore()
    state = store.create("conv-refresh-failure")
    state.phase = ConversationPhase.CONFIRMING_BOOKING
    state.verified_customer = customer
    state.selected_pet_id = "pet_2001"
    state.appointment_type = AppointmentType.SAME_DAY
    state.selected_slot_id = "slot-old"
    store.save(state)

    class RefreshFailureScheduler(FakeSchedulerClient):
        async def book_appointment(self, booking_request: BookingRequest, idempotency_key: str) -> BookingConfirmation:
            raise SlotUnavailableError("slot unavailable")

        async def find_slots(self, pet_id: str, appointment_type: AppointmentType) -> list[AppointmentSlot]:
            raise SchedulerRequestError("refresh failed")

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
        scheduler_client=RefreshFailureScheduler(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )

    with pytest.raises(AgentToolError, match="slot refresh"):
        await orchestrator.handle_turn("conv-refresh-failure", "Yes, book it.")


@pytest.mark.asyncio
async def test_different_conversation_ids_can_interpret_concurrently() -> None:
    store = InMemoryConversationStore()
    entered = {"conv-a": asyncio.Event(), "conv-b": asyncio.Event()}
    release = asyncio.Event()

    class ConcurrentInterpreter(FakeTurnInterpreter):
        async def interpret(self, state: ConversationState, user_message: str) -> TurnUnderstanding:
            entered[state.conversation_id].set()
            await release.wait()
            return await super().interpret(state, user_message)

    orchestrator = AgentOrchestrator(
        conversation_store=store,
        turn_interpreter=ConcurrentInterpreter(),
        legacy_crm_client=FakeLegacyCRMClient(None),
        scheduler_client=FakeSchedulerClient(),
        handoff_client=FakeHandoffClient(),
        policy_service=FakePolicyService(),
        idempotency_key_factory=lambda: "key-1",
    )
    await orchestrator.start_conversation("conv-a")
    await orchestrator.start_conversation("conv-b")

    first = asyncio.create_task(orchestrator.handle_turn("conv-a", "first"))
    second = asyncio.create_task(orchestrator.handle_turn("conv-b", "second"))
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), timeout=1)
    release.set()
    await asyncio.gather(first, second)


@pytest.mark.asyncio
async def test_same_conversation_turns_are_serialized_by_lock() -> None:
    store = InMemoryConversationStore()
    first_interpreter_entered = asyncio.Event()
    second_turn_started = asyncio.Event()
    release_first_turn = asyncio.Event()

    class BlockingInterpreter(FakeTurnInterpreter):
        async def interpret(self, state: ConversationState, user_message: str):
            if user_message == "first message":
                first_interpreter_entered.set()
                await release_first_turn.wait()
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
    await first_interpreter_entered.wait()

    async def send_second_turn() -> None:
        second_turn_started.set()
        await orchestrator.handle_turn("conv-lock", "second message")

    second = asyncio.create_task(send_second_turn())
    await second_turn_started.wait()
    assert not second.done()
    release_first_turn.set()
    await asyncio.gather(first, second)
    assert store.get("conv-lock").turn_count == 2
