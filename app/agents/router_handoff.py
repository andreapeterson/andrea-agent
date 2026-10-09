"""Use the router's decision to send a caller message to an implemented specialist."""

from __future__ import annotations

from app.agents.prompt_context import PromptContext
from app.agents.policy_agent import PolicyAgent
from app.agents.prompt_renderer import PromptRenderer
from app.agents.router_agent import OpenAIRouter, RouteDecision
from app.agents.verification_agent import VerificationAgent


class SpecialistNotAvailableError(RuntimeError):
    """Raised when the router selects a specialist that has not been implemented."""


def route_and_respond(
    *,
    router: OpenAIRouter,
    renderer: PromptRenderer,
    verification_agent: VerificationAgent,
    policy_agent: PolicyAgent | None = None,
) -> object:
    """Render the router prompt, route the message, and call the implemented specialist.

    Verification and a supplied Policy Agent are supported. Other destinations
    remain unavailable until their agents are implemented.
    """

    async def handle(context: PromptContext, user_message: str) -> str:
        instructions = renderer.render_router(context, user_message)
        decision = await router.route(
            instructions=instructions,
            user_message=user_message,
        )

        if decision.destination == "verification":
            return await verification_agent.respond(
                context=context,
                user_message=user_message,
            )

        if decision.destination == "policy" and policy_agent is not None:
            return await policy_agent.respond(
                context=context,
                user_message=user_message,
            )

        raise SpecialistNotAvailableError(
            f"The specialist '{decision.destination}' is not available."
        )

    return handle


__all__ = ["SpecialistNotAvailableError", "route_and_respond"]
