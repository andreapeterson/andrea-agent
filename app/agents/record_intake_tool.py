"""Record caller-supplied intake facts and assess the existing fictional routing rules."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from app.models.conversation import ConversationState
from app.services.conversation_store import ConversationStore
from app.services.routing import assess_routing


class RecordIntakeArguments(BaseModel):
    """All fields are required; null means no new information in this message."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", strict=True)

    original_concern: str | None = Field(min_length=1, description="Caller-stated concern, or null if none is supplied.")
    difficulty_breathing: bool | None
    uncontrolled_bleeding: bool | None
    collapsed_or_unresponsive: bool | None
    known_toxin_exposure: bool | None
    rapidly_worsening: bool | None


RECORD_INTAKE_TOOL: dict[str, object] = {
    "type": "function",
    "name": "record_intake",
    "description": (
        "Record only the concern and intake answers the caller actually supplied. "
        "Use null for each field with no new information; false means an explicit negative answer. "
        "Returns the current recorded answers and the application's fictional routing decision. "
        "It does not diagnose, book an appointment, or transfer the caller."
    ),
    "strict": True,
    "parameters": RecordIntakeArguments.model_json_schema(),
}


def create_record_intake_handler(
    *,
    conversation_state: ConversationState,
    conversation_store: ConversationStore,
) -> Callable[[dict[str, object]], Awaitable[object]]:
    """Prepare a recording tool for one conversation, using its existing state and store."""

    async def handler(arguments: dict[str, object]) -> object:
        validated = RecordIntakeArguments.model_validate(arguments)

        if conversation_state.original_concern is None and validated.original_concern is not None:
            conversation_state.original_concern = validated.original_concern

        updates = validated.model_dump(exclude={"original_concern"})
        for field_name, value in updates.items():
            # Keep an existing true answer so a later tool call cannot clear an urgent signal.
            if value is not None and getattr(conversation_state.intake_answers, field_name) is not True:
                setattr(conversation_state.intake_answers, field_name, value)

        decision = assess_routing(conversation_state.intake_answers)
        conversation_state.routing_decision = decision
        conversation_store.save(conversation_state)

        # The prompt was rendered before this tool ran; return the newly recorded facts.
        return {
            "original_concern": conversation_state.original_concern,
            "intake_answers": conversation_state.intake_answers.model_dump(mode="json"),
            **decision.model_dump(mode="json"),
        }

    return handler


__all__ = ["RECORD_INTAKE_TOOL", "RecordIntakeArguments", "create_record_intake_handler"]
