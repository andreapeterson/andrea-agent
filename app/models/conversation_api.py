"""Public request and response models for multi-turn conversation API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.agent_result import AgentTurnResult
from app.models.appointment import AppointmentSlot, BookingConfirmation
from app.models.conversation import ConversationPhase
from app.models.handoff import HandoffReceipt
from app.models.policy_answer import PolicyAnswerResponse


class ConversationTurnRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    message: str = Field(..., min_length=1)


class ConversationResponse(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    conversation_id: str
    phase: ConversationPhase
    assistant_message: str
    turn_count: int
    available_slots: list[AppointmentSlot] = Field(default_factory=list)
    policy_answer: PolicyAnswerResponse | None = None
    booking_confirmation: BookingConfirmation | None = None
    handoff_receipt: HandoffReceipt | None = None


def conversation_response_from_result(result: AgentTurnResult) -> ConversationResponse:
    state = result.state
    available_slots: list[AppointmentSlot] = []
    if state.phase == ConversationPhase.SELECTING_APPOINTMENT and state.offered_slots:
        available_slots = list(state.offered_slots)

    return ConversationResponse(
        conversation_id=state.conversation_id,
        phase=state.phase,
        assistant_message=result.assistant_message,
        turn_count=state.turn_count,
        available_slots=available_slots,
        policy_answer=result.policy_answer,
        booking_confirmation=state.booking_confirmation,
        handoff_receipt=state.handoff_receipt,
    )