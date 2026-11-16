"""Trusted state gates available specialists; the Router still interprets caller intent."""

from app.agents.router_destinations import ROUTER_DESTINATIONS, RouterDestination
from app.models.conversation import ConversationPhase, ConversationState
from app.models.routing import RoutingLevel
from app.services.routing import assess_routing


def eligible_specialists(state: ConversationState) -> tuple[RouterDestination, ...]:
    """Return stable, immutable choices without inspecting messages or modifying state."""
    if state.phase in {
        ConversationPhase.BOOKING_COMPLETE,
        ConversationPhase.HANDOFF_COMPLETE,
        ConversationPhase.COMPLETED,
    }:
        return ()

    assessment = assess_routing(state.intake_answers)
    if assessment.routing_level == RoutingLevel.URGENT:
        return ("human_handoff",)

    allowed: set[RouterDestination] = {"intake", "policy", "human_handoff"}
    if state.verified_customer is None:
        allowed.add("verification")
    elif (
        state.original_concern and state.original_concern.strip()
        and assessment.routing_level in {RoutingLevel.ROUTINE, RoutingLevel.SAME_DAY}
        and state.booking_confirmation is None
        and state.handoff_receipt is None
    ):
        # Pending booking keys remain intact and scheduling owns their recovery.
        allowed.add("scheduling")
    return tuple(destination for destination in ROUTER_DESTINATIONS if destination in allowed)
