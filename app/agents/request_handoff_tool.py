"""Queue human assistance from trusted conversation state through HandoffClient."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.integrations import (
    HandoffClient,
    HandoffNotRequiredError,
    HandoffRequestError,
    HandoffResponseError,
)
from app.models.conversation import ConversationPhase, ConversationState
from app.models.handoff import HandoffReason, HandoffReceipt, HandoffRequest
from app.models.routing import RoutingLevel
from app.services.conversation_store import ConversationStore
from app.services.handoff_summary import build_handoff_summary
from app.services.routing import assess_routing


class RequestHandoffArguments(BaseModel):
    """The model proposes only a reason and explanation, never identity or clinical data."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    reason: HandoffReason
    explanation: str = Field(min_length=1, description="Why human assistance is requested.")


REQUEST_HANDOFF_TOOL: dict[str, object] = {
    "type": "function",
    "name": "request_handoff",
    "description": (
        "Request human assistance for an explicit caller request, an out-of-scope request, "
        "a safety concern, or urgent recorded intake. Python uses trusted conversation state "
        "and reassesses urgency. Success means the request is queued, not a live transfer."
    ),
    "strict": True,
    "parameters": RequestHandoffArguments.model_json_schema(),
}


def create_request_handoff_handler(
    handoff_client: HandoffClient,
    *,
    conversation_state: ConversationState,
    conversation_store: ConversationStore,
) -> Callable[[dict[str, object]], Awaitable[object]]:
    """Prepare one conversation's handoff function for use by either specialist."""

    async def handler(arguments: dict[str, object]) -> object:
        validated = RequestHandoffArguments.model_validate(arguments)
        if conversation_state.handoff_receipt is not None:
            return {"queued": True, "case_id": conversation_state.handoff_receipt.case_id}

        decision = assess_routing(conversation_state.intake_answers)
        urgent = decision.routing_level == RoutingLevel.URGENT
        if validated.reason == HandoffReason.URGENT_INTAKE and not urgent: #if model is requesting an urgent intake handoff, but the recorded answers aren’t urgent, reject it
            return {"queued": False, "case_id": None, "error": "handoff-not-required"}

        reason = HandoffReason.URGENT_INTAKE if urgent else validated.reason #if recorded answers are urgent, override the model's reason to be URGENT_INTAKE
        concern = (conversation_state.original_concern or "").strip() or None
        summary = (
            build_handoff_summary(concern, conversation_state.intake_answers, decision)
            if urgent and concern is not None
            else None
        )
        request = HandoffRequest(
            conversation_id=conversation_state.conversation_id,
            reason=reason,
            explanation=validated.explanation,
            verified_customer_id=(
                conversation_state.verified_customer.customer_id
                if conversation_state.verified_customer is not None
                else conversation_state.verified_customer_id
            ),
            selected_pet_id=conversation_state.selected_pet_id,
            original_concern=concern if urgent else None,
            intake_answers=conversation_state.intake_answers if urgent else None,
            routing_decision=decision if urgent else None,
            summary=summary,
        )

        try:
            receipt = await handoff_client.create_handoff(request)
            receipt = HandoffReceipt.model_validate(receipt)
            if receipt.conversation_id != conversation_state.conversation_id:
                raise HandoffResponseError("Receipt conversation does not match the request.")
        except HandoffNotRequiredError:
            return {"queued": False, "case_id": None, "error": "handoff-not-required"}
        except (HandoffResponseError, ValidationError):
            return {"queued": False, "case_id": None, "error": "invalid-handoff-response"}
        except HandoffRequestError:
            return {"queued": False, "case_id": None, "error": "handoff-service-unavailable"}

        # Only an accepted, validated receipt marks the conversation's handoff complete.
        conversation_state.handoff_receipt = receipt
        conversation_state.phase = ConversationPhase.HANDOFF_COMPLETE
        if urgent:
            conversation_state.routing_decision = decision
        conversation_store.save(conversation_state)
        return {"queued": True, "case_id": receipt.case_id}

    return handler


__all__ = ["REQUEST_HANDOFF_TOOL", "RequestHandoffArguments", "create_request_handoff_handler"]
