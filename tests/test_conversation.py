from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models.appointment import AppointmentSlot, AppointmentType, BookingConfirmation, BookingStatus
from app.models.conversation import (
    ConversationMessage,
    ConversationPhase,
    ConversationRole,
    ConversationState,
)
from app.models.customer import Customer, Pet, PetSpecies
from app.models.handoff import HandoffReceipt, HandoffStatus
from app.models.routing import IntakeAnswers, RoutingAction, RoutingDecision, RoutingLevel
from app.services.conversation_store import (
    ConversationAlreadyExistsError,
    ConversationNotFoundError,
    InMemoryConversationStore,
)


def test_new_state_starts_in_started_phase_and_has_no_customer_or_pet() -> None:
    state = ConversationState(conversation_id="conv-001")

    assert state.phase is ConversationPhase.STARTED
    assert state.verified_customer is None
    assert state.selected_pet_id is None
    assert state.messages == []
    assert state.offered_slots == []
    assert state.intake_answers.difficulty_breathing is None
    assert state.intake_answers.uncontrolled_bleeding is None
    assert state.intake_answers.collapsed_or_unresponsive is None
    assert state.intake_answers.known_toxin_exposure is None
    assert state.intake_answers.rapidly_worsening is None


def test_separate_states_do_not_share_lists_or_intake_objects() -> None:
    state_one = ConversationState(conversation_id="conv-001")
    state_two = ConversationState(conversation_id="conv-002")

    state_one.messages.append(ConversationMessage(role=ConversationRole.USER, content="hello"))
    state_one.offered_slots.append(
        AppointmentSlot(
            slot_id="slot-01",
            clinic_id="clinic-01",
            starts_at=datetime(2026, 9, 27, 9, 0, 0),
            ends_at=datetime(2026, 9, 27, 9, 30, 0),
            appointment_type=AppointmentType.SAME_DAY,
        )
    )
    state_one.intake_answers.difficulty_breathing = True

    assert state_two.messages == []
    assert state_two.offered_slots == []
    assert state_two.intake_answers.difficulty_breathing is None
    assert state_one.intake_answers is not state_two.intake_answers


def test_conversation_message_strips_whitespace_and_rejects_blank_content() -> None:
    message = ConversationMessage(role=ConversationRole.ASSISTANT, content="  hello there  ")
    assert message.content == "hello there"

    with pytest.raises(ValidationError):
        ConversationMessage(role=ConversationRole.USER, content="   ")


def test_conversation_model_rejects_extra_fields_and_negative_turn_count() -> None:
    with pytest.raises(ValidationError):
        ConversationMessage.model_validate({"role": "user", "content": "hello", "extra": "nope"})

    with pytest.raises(ValidationError):
        ConversationState(conversation_id="conv-001", turn_count=-1)


def test_conversation_state_can_hold_existing_domain_models() -> None:
    customer = Customer(
        customer_id="customer_1001",
        first_name="Andrea",
        last_name="Peterson",
        phone_number="3212222222",
        pets=[Pet(pet_id="pet_2001", name="Milo", species=PetSpecies.DOG)],
    )
    slot = AppointmentSlot(
        slot_id="slot-01",
        clinic_id="clinic-01",
        starts_at=datetime(2026, 9, 27, 10, 0, 0),
        ends_at=datetime(2026, 9, 27, 10, 30, 0),
        appointment_type=AppointmentType.ROUTINE,
    )
    routing_decision = RoutingDecision(
        routing_level=RoutingLevel.SAME_DAY,
        next_action=RoutingAction.SEARCH_SAME_DAY_APPOINTMENT,
        matched_rule_ids=["rule_1"],
        missing_fields=[],
    )
    booking_confirmation = BookingConfirmation(
        booking_id="booking-001",
        slot_id="slot-01",
        pet_id="pet_2001",
        status=BookingStatus.CONFIRMED,
    )
    handoff_receipt = HandoffReceipt(
        case_id="case-001",
        conversation_id="conv-001",
        status=HandoffStatus.QUEUED,
        received_at=datetime(2026, 9, 27, 12, 0, 0),
    )

    state = ConversationState(
        conversation_id="conv-001",
        verified_customer=customer,
        selected_pet_id="pet_2001",
        original_concern="labored breathing",
        intake_answers=IntakeAnswers(difficulty_breathing=True, rapidly_worsening=False),
        routing_decision=routing_decision,
        appointment_type=AppointmentType.SAME_DAY,
        offered_slots=[slot],
        selected_slot_id="slot-01",
        booking_confirmation=booking_confirmation,
        handoff_receipt=handoff_receipt,
        turn_count=4,
    )

    assert state.verified_customer.customer_id == "customer_1001"
    assert state.offered_slots[0].slot_id == "slot-01"
    assert state.booking_confirmation.booking_id == "booking-001"
    assert state.routing_decision.next_action is RoutingAction.SEARCH_SAME_DAY_APPOINTMENT
    assert state.handoff_receipt.case_id == "case-001"


def test_in_memory_conversation_store_creates_gets_and_saves_state() -> None:
    store = InMemoryConversationStore()

    created = store.create("conv-001")
    assert created.conversation_id == "conv-001"
    assert created.phase is ConversationPhase.STARTED

    retrieved = store.get("conv-001")
    assert retrieved.conversation_id == "conv-001"

    retrieved.phase = ConversationPhase.SELECTING_PET
    retrieved.turn_count = 2
    store.save(retrieved)

    saved = store.get("conv-001")
    assert saved.phase is ConversationPhase.SELECTING_PET
    assert saved.turn_count == 2


def test_in_memory_store_rejects_duplicate_blank_and_missing_conversations() -> None:
    store = InMemoryConversationStore()

    with pytest.raises(ValueError):
        store.create("   ")

    store.create("conv-001")
    with pytest.raises(ConversationAlreadyExistsError):
        store.create("conv-001")

    with pytest.raises(ConversationNotFoundError):
        store.get("conv-404")

    with pytest.raises(ConversationNotFoundError):
        store.save(ConversationState(conversation_id="conv-404"))


def test_in_memory_store_returns_deep_copies_for_create_get_and_save() -> None:
    store = InMemoryConversationStore()
    created = store.create("conv-001")
    created.messages.append(ConversationMessage(role=ConversationRole.USER, content="hello"))
    created.intake_answers.difficulty_breathing = True
    created.offered_slots.append(
        AppointmentSlot(
            slot_id="slot-01",
            clinic_id="clinic-01",
            starts_at=datetime(2026, 9, 27, 9, 0, 0),
            ends_at=datetime(2026, 9, 27, 9, 30, 0),
            appointment_type=AppointmentType.SAME_DAY,
        )
    )

    stored = store.get("conv-001")
    assert stored.messages == []
    assert stored.intake_answers.difficulty_breathing is None
    assert stored.offered_slots == []

    updated = store.get("conv-001")
    updated.phase = ConversationPhase.COLLECTING_INTAKE
    updated.turn_count = 3
    store.save(updated)

    persisted = store.get("conv-001")
    assert persisted.phase is ConversationPhase.COLLECTING_INTAKE
    assert persisted.turn_count == 3


def test_store_mutations_remain_isolated_between_conversations() -> None:
    store = InMemoryConversationStore()
    state_one = store.create("conv-001")
    state_two = store.create("conv-002")

    state_one.messages.append(ConversationMessage(role=ConversationRole.USER, content="first"))
    state_one.intake_answers.difficulty_breathing = True
    state_two.messages.append(ConversationMessage(role=ConversationRole.ASSISTANT, content="second"))
    state_two.intake_answers.collapsed_or_unresponsive = False

    assert not store.get("conv-001").messages
    assert store.get("conv-001").intake_answers.difficulty_breathing is None
    assert not store.get("conv-002").messages == [ConversationMessage(role=ConversationRole.USER, content="first")]
    assert store.get("conv-002").intake_answers.collapsed_or_unresponsive is None
