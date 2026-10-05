"""Define the conversation facts that may be copied into an agent prompt."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.appointment import AppointmentSlot
from app.models.conversation import ConversationMessage
from app.models.routing import IntakeAnswers, RoutingDecision


class PromptContext(BaseModel):
    """A temporary collection of conversation facts that will be placed into an
    agent's prompt.

    PawLine copies these values from ConversationState and other trusted
    application data before each LLM call. Only fields defined here may be added
    to the prompt. This object does not store the conversation, call the LLM, or
    decide what the agent should do.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    active_agent: str = "front_desk"
    customer_verified: bool | None = None
    customer_first_name: str | None = None
    known_pets: list[str] = Field(default_factory=list)
    selected_pet: str | None = None
    original_concern: str | None = None
    intake_answers: IntakeAnswers = Field(default_factory=IntakeAnswers)
    missing_intake_answers: list[str] = Field(default_factory=list)
    routing_result: RoutingDecision | None = None
    offered_slots: list[AppointmentSlot] = Field(default_factory=list)
    selected_slot: AppointmentSlot | None = None
    booking_confirmed: bool | None = None
    recent_messages: list[ConversationMessage] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)


__all__ = ["PromptContext"]