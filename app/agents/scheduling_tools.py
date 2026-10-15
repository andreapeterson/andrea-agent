"""Find and select owned-pet appointment slots without making a booking."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.integrations.scheduler_client import SchedulerClient, SchedulerRequestError, SchedulerResponseError
from app.models.appointment import AppointmentSlot, AppointmentType
from app.models.conversation import ConversationPhase, ConversationState
from app.models.customer import Pet
from app.models.routing import RoutingLevel
from app.services.conversation_store import ConversationStore
from app.services.routing import assess_routing


class FindAppointmentSlotsArguments(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", strict=True)

    pet_reference: str | None = Field(
        min_length=1,
        description="Caller-provided pet name or ID, or null to reuse the selected pet.",
    )


class SelectAppointmentSlotArguments(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", strict=True)

    slot_id: str = Field(min_length=1, description="Exact ID of a slot offered in this conversation.")


FIND_APPOINTMENT_SLOTS_TOOL: dict[str, object] = {
    "type": "function",
    "name": "find_appointment_slots",
    "description": (
        "Find available appointment slots for a verified customer's pet after completed non-urgent intake. "
        "Use the caller-provided pet reference, or null to reuse a selected pet. "
        "Python determines the appointment type from recorded intake. This tool does not book."
    ),
    "strict": True,
    "parameters": FindAppointmentSlotsArguments.model_json_schema(),
}

SELECT_APPOINTMENT_SLOT_TOOL: dict[str, object] = {
    "type": "function",
    "name": "select_appointment_slot",
    "description": (
        "Select an exact slot ID from this conversation's offered appointments. "
        "Returns the selected appointment for caller confirmation. Selection is not a booking."
    ),
    "strict": True,
    "parameters": SelectAppointmentSlotArguments.model_json_schema(),
}


def _scheduling_eligibility(state: ConversationState) -> tuple[AppointmentType | None, dict[str, object] | None]:
    """Read trusted state; keep blocked and pending attempts completely unchanged."""
    if state.handoff_receipt is not None or state.phase == ConversationPhase.HANDOFF_COMPLETE:
        return None, {"status": "handoff_complete"}
    if state.booking_confirmation is not None or state.phase == ConversationPhase.BOOKING_COMPLETE:
        return None, {"status": "booking_complete"}
    if state.phase == ConversationPhase.COMPLETED:
        return None, {"status": "conversation_complete"}
    if state.booking_idempotency_key is not None:
        return None, {"status": "booking_recovery_required"}
    # An ID or prompt flag alone does not establish a CRM-confirmed customer.
    if state.verified_customer is None:
        return None, {"status": "verification_required"}

    decision = assess_routing(state.intake_answers)
    if decision.routing_level == RoutingLevel.URGENT:
        return None, {"status": "handoff_required"}
    if decision.routing_level == RoutingLevel.NEEDS_MORE_INFORMATION:
        return None, {"status": "intake_required", "missing_fields": decision.missing_fields}
    appointment_type = (
        AppointmentType.SAME_DAY
        if decision.routing_level == RoutingLevel.SAME_DAY
        else AppointmentType.ROUTINE
    )
    return appointment_type, None


def _resolve_pet(state: ConversationState, reference: str | None) -> Pet | None:
    pets = state.verified_customer.pets
    if reference is not None:
        matches = [pet for pet in pets if pet.pet_id == reference]
        if not matches:
            matches = [pet for pet in pets if pet.name.casefold() == reference.casefold()]
        return matches[0] if len(matches) == 1 else None
    selected = [pet for pet in pets if pet.pet_id == state.selected_pet_id]
    if len(selected) == 1:
        return selected[0]
    return pets[0] if len(pets) == 1 else None


def _pet_details(pet: Pet) -> dict[str, str]:
    return {"pet_id": pet.pet_id, "name": pet.name}


def _pet_clarification(state: ConversationState) -> dict[str, object]:
    return {
        "status": "pet_clarification_required",
        "pets": [_pet_details(pet) for pet in state.verified_customer.pets],
    }


def _validated_slots(values: object, appointment_type: AppointmentType) -> list[AppointmentSlot]:
    if not isinstance(values, list):
        raise SchedulerResponseError("Expected a list of slots.")
    slots = [
        AppointmentSlot.model_validate(value.model_dump() if isinstance(value, AppointmentSlot) else value)
        for value in values
    ]
    ids = [slot.slot_id for slot in slots]
    if len(ids) != len(set(ids)) or any(
        not slot.slot_id or not slot.clinic_id or slot.appointment_type != appointment_type
        for slot in slots
    ):
        raise SchedulerResponseError("Slot identifiers or appointment types are invalid.")
    return slots


def create_find_appointment_slots_handler(
    scheduler_client: SchedulerClient,
    *,
    conversation_state: ConversationState,
    conversation_store: ConversationStore,
) -> Callable[[dict[str, object]], Awaitable[object]]:
    """Prepare a search handler using this conversation's trusted state and store."""

    async def handler(arguments: dict[str, object]) -> object:
        validated = FindAppointmentSlotsArguments.model_validate(arguments)
        appointment_type, blocked = _scheduling_eligibility(conversation_state)
        if blocked is not None:
            return blocked
        pet = _resolve_pet(conversation_state, validated.pet_reference)
        if pet is None:
            return _pet_clarification(conversation_state)
        customer_id = conversation_state.verified_customer.customer_id

        try:
            values = await scheduler_client.find_slots(pet.pet_id, appointment_type)
            slots = _validated_slots(values, appointment_type)
        except SchedulerRequestError:
            return {"status": "scheduler_unavailable"}
        except (SchedulerResponseError, ValidationError, TypeError):
            return {"status": "invalid_scheduler_response"}

        # State may change while waiting for the scheduler; never overwrite a pending booking or handoff. i.e. if request b is urgent handoff dont overwrite it with request a's non-urgent slots
        current_type, blocked = _scheduling_eligibility(conversation_state)
        if blocked is not None:
            return blocked
        if current_type != appointment_type or conversation_state.verified_customer.customer_id != customer_id:
            return {"status": "search_required"}
        if not any(owned.pet_id == pet.pet_id for owned in conversation_state.verified_customer.pets):
            return _pet_clarification(conversation_state)

        conversation_state.selected_pet_id = pet.pet_id
        conversation_state.appointment_type = appointment_type
        conversation_state.routing_decision = assess_routing(conversation_state.intake_answers)
        conversation_state.offered_slots = slots
        conversation_state.selected_slot_id = None
        conversation_state.phase = ConversationPhase.SELECTING_APPOINTMENT
        conversation_store.save(conversation_state)
        return {
            "status": "slots_found" if slots else "no_slots",
            "selected_pet": _pet_details(pet),
            "appointment_type": appointment_type.value,
            "slots": [slot.model_dump(mode="json") for slot in slots],
        }

    return handler


def create_select_appointment_slot_handler(
    *,
    conversation_state: ConversationState,
    conversation_store: ConversationStore,
) -> Callable[[dict[str, object]], Awaitable[object]]:
    """Prepare selection of an offered slot; this handler has no booking client."""

    async def handler(arguments: dict[str, object]) -> object:
        validated = SelectAppointmentSlotArguments.model_validate(arguments)
        appointment_type, blocked = _scheduling_eligibility(conversation_state)
        if blocked is not None:
            return blocked
        pets = [
            pet for pet in conversation_state.verified_customer.pets
            if pet.pet_id == conversation_state.selected_pet_id
        ]
        if len(pets) != 1:
            return _pet_clarification(conversation_state)
        if conversation_state.appointment_type != appointment_type:
            return {"status": "search_required"}
        try:
            slots = _validated_slots(conversation_state.offered_slots, appointment_type)
        except (SchedulerResponseError, ValidationError, TypeError):
            return {"status": "search_required"}
        matches = [slot for slot in slots if slot.slot_id == validated.slot_id]
        if not matches:
            return {"status": "slot_not_offered"}

        slot = matches[0]
        conversation_state.selected_slot_id = slot.slot_id
        conversation_state.routing_decision = assess_routing(conversation_state.intake_answers)
        conversation_state.phase = ConversationPhase.CONFIRMING_BOOKING
        conversation_store.save(conversation_state)
        return {
            "status": "slot_selected",
            "selected_pet": _pet_details(pets[0]),
            "slot": slot.model_dump(mode="json"),
            "confirmation_required": True,
        }

    return handler


__all__ = [
    "FIND_APPOINTMENT_SLOTS_TOOL", "SELECT_APPOINTMENT_SLOT_TOOL",
    "FindAppointmentSlotsArguments", "SelectAppointmentSlotArguments",
    "create_find_appointment_slots_handler", "create_select_appointment_slot_handler",
]
