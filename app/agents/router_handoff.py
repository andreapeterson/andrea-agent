"""Use the router's decision to send a caller message to an implemented specialist."""

from __future__ import annotations

from collections.abc import Callable

from app.agents.human_handoff_agent import HumanHandoffAgent
from app.agents.intake_agent import IntakeAgent
from app.agents.policy_agent import PolicyAgent
from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import (
    HUMAN_HANDOFF_PROMPT_VERSION,
    INTAKE_PROMPT_VERSION,
    POLICY_PROMPT_VERSION,
    SCHEDULING_PROMPT_VERSION,
    VERIFICATION_PROMPT_VERSION,
    PromptRenderer,
)
from app.agents.router_agent import Router, RouterDestination, RouterResponseError
from app.agents.router_destinations import validate_eligible_destinations
from app.agents.scheduling_agent import SchedulingAgent
from app.agents.verification_agent import VerificationAgent
from app.agents.workflow_observation import observe_specialist

SPECIALIST_PROMPT_VERSIONS = {
    "verification": VERIFICATION_PROMPT_VERSION,
    "policy": POLICY_PROMPT_VERSION,
    "intake": INTAKE_PROMPT_VERSION,
    "scheduling": SCHEDULING_PROMPT_VERSION,
    "human_handoff": HUMAN_HANDOFF_PROMPT_VERSION,
}


class SpecialistNotAvailableError(RuntimeError):
    """Raised when the router selects a specialist that has not been implemented."""


def route_and_respond(
    *,
    router: Router,
    renderer: PromptRenderer,
    eligible_destinations: tuple[RouterDestination, ...],
    verification_agent: VerificationAgent | None = None,
    policy_agent: PolicyAgent | None = None,
    intake_agent: IntakeAgent | None = None,
    human_handoff_agent: HumanHandoffAgent | None = None,
    scheduling_agent: SchedulingAgent | None = None,
    specialist_factory: Callable[
        [RouterDestination],
        VerificationAgent | PolicyAgent | IntakeAgent | HumanHandoffAgent | SchedulingAgent | None,
    ] | None = None,
) -> object:
    """Render the router prompt, route the message, and call the implemented specialist.

    Verification and supplied Policy, Intake, Human Handoff, and Scheduling Agents
    are supported. An unsupplied specialist remains unavailable.
    """

    choices = validate_eligible_destinations(eligible_destinations)

    async def handle(context: PromptContext, user_message: str) -> str:
        instructions = renderer.render_router(context, eligible_destinations=choices)
        decision = await router.route(
            instructions=instructions,
            user_message=user_message,
            eligible_destinations=choices,
        )
        if decision.destination not in choices:
            raise RouterResponseError("Router selected an ineligible specialist.")
        observe_specialist(
            decision.destination, prompt_version=SPECIALIST_PROMPT_VERSIONS.get(decision.destination),
        )

        # Build handlers only after validation, including when the Router is faulty.
        agent = specialist_factory(decision.destination) if specialist_factory else {
            "verification": verification_agent,
            "intake": intake_agent,
            "policy": policy_agent,
            "scheduling": scheduling_agent,
            "human_handoff": human_handoff_agent,
        }.get(decision.destination)
        if agent is not None:
            return await agent.respond(context=context, user_message=user_message)

        raise SpecialistNotAvailableError(
            f"The specialist '{decision.destination}' is not available."
        )

    return handle


__all__ = ["SpecialistNotAvailableError", "route_and_respond"]
