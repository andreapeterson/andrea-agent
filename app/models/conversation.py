"""Conversation-state models for PawLine's future multi-turn agent memory."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .appointment import AppointmentSlot, AppointmentType, BookingConfirmation
from .customer import Customer
from .handoff import HandoffReceipt
from .routing import IntakeAnswers, RoutingDecision


class ConversationRole(str, Enum):
    USER = "user"
    
    ASSISTANT = "assistant"


class ConversationPhase(str, Enum):
    STARTED = "started"
    VERIFYING_CUSTOMER = "verifying_customer"
    SELECTING_PET = "selecting_pet"
    COLLECTING_CONCERN = "collecting_concern"
    COLLECTING_INTAKE = "collecting_intake"
    SELECTING_APPOINTMENT = "selecting_appointment"
    CONFIRMING_BOOKING = "confirming_booking"
    HANDOFF_COMPLETE = "handoff_complete"
    BOOKING_COMPLETE = "booking_complete"
    COMPLETED = "completed"


class ConversationMessage(BaseModel):
    """A single user/assistant message in a conversation snapshot."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    role: ConversationRole
    content: str = Field(..., min_length=1)


class ConversationState(BaseModel):
    """Internal agent memory for one conversation thread."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    conversation_id: str = Field(..., min_length=1)
    phase: ConversationPhase = ConversationPhase.STARTED
    messages: list[ConversationMessage] = Field(default_factory=list)
    verified_customer: Customer | None = None
    verified_customer_id: str | None = None
    selected_pet_id: str | None = None
    original_concern: str | None = None
    intake_answers: IntakeAnswers = Field(default_factory=IntakeAnswers)
    routing_decision: RoutingDecision | None = None
    appointment_type: AppointmentType | None = None
    offered_slots: list[AppointmentSlot] = Field(default_factory=list)
    selected_slot_id: str | None = None
    booking_idempotency_key: str | None = None
    booking_confirmation: BookingConfirmation | None = None
    handoff_receipt: HandoffReceipt | None = None
    turn_count: int = Field(default=0, ge=0)


__all__ = [
    "ConversationMessage",
    "ConversationPhase",
    "ConversationRole",
    "ConversationState",
]
