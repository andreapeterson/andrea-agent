"""Opt-in real conversation workflows backed only by controlled demo integrations."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI

from app.agents.policy_agent import PolicyAgent
from app.agents.prompt_renderer import PromptRenderer
from app.agents.search_policy_tool import create_search_policy_handler
from app.agents.workflow_observation import WorkflowEvent, WorkflowEventType
from app.models.appointment import AppointmentSlot, AppointmentType, BookingConfirmation, BookingRequest
from app.models.customer import Customer, Pet
from app.models.handoff import HandoffReceipt, HandoffRequest
from app.models.policy import PolicyChunk
from app.models.retrieval import PolicySearchResult
from app.services.conversation_store import InMemoryConversationStore
from app.services.prompt_agent_runtime import PromptAgentRuntime
from evals.failure_attribution import attribute_report_failures, format_failure_attribution_report
from evals.conversation_eval import (
    ConversationEvalCase,
    ConversationRunObservation,
    format_conversation_report,
    load_conversation_cases,
    run_conversation_evaluation,
)

DEMO_PHONES = ("202-555-0142", "202-555-0143")
DEMO_PET_IDS = {f"demo-pet-maize-{phone[-4:]}" for phone in DEMO_PHONES}


class LiveConversationEvaluationError(RuntimeError):
    """A workflow could not produce a valid observation; never expose upstream errors."""


class ControlledIntegrationError(RuntimeError):
    """The model requested an operation outside the controlled demo fixtures."""


class DemoCRM:
    async def find_customer_by_phone(self, phone: str) -> Customer | None:
        if phone not in DEMO_PHONES:
            return None
        suffix = phone[-4:]
        return Customer(
            customer_id=f"demo-customer-{suffix}", first_name="Avery", last_name="Demo",
            phone_number=phone,
            pets=[Pet(pet_id=f"demo-pet-maize-{suffix}", name="Maize", species="dog")],
        )


class DemoPolicyRetriever:
    async def search(self, query: str, limit: int = 3) -> list[PolicySearchResult]:
        if not query.strip() or limit < 1:
            raise ControlledIntegrationError("Invalid demo policy search")
        if not any(word in query.casefold() for word in ("hour", "open", "close")):
            return [] #so policy agent can perform real tool calling without using embeddings or a real database
        return [PolicySearchResult(
            chunk=PolicyChunk(
                chunk_id="demo-hours-1", document_id="demo-clinic-policy",
                document_title="Demo Clinic Policy", section_title="Clinic Hours",
                source_name="Demo clinic policy",
                text="The clinic is open Monday through Friday, 8:00 a.m. to 6:00 p.m. Eastern Time.",
            ), similarity_score=1.0,
        )]


class DemoScheduler:
    def __init__(self) -> None:
        self._bookings: dict[str, tuple[BookingRequest, BookingConfirmation]] = {}
        self._booked_slots: set[str] = set()

    def _slots(self, appointment_type: AppointmentType) -> list[AppointmentSlot]:
        #two booking slots per appointment type, with a 30-minute duration, in the demo clinic timezone
        return [AppointmentSlot(
            slot_id=f"demo-{appointment_type.value}-slot-{index}", clinic_id="demo-clinic",
            starts_at=datetime(2026, 11, 3, hour, tzinfo=timezone(timedelta(hours=-5))),
            ends_at=datetime(2026, 11, 3, hour, 30, tzinfo=timezone(timedelta(hours=-5))),
            appointment_type=appointment_type,
        ) for index, hour in enumerate((9, 10), start=1)]

    async def find_slots(self, pet_id: str, appointment_type: AppointmentType) -> list[AppointmentSlot]:
        # returns available slots
        if pet_id not in DEMO_PET_IDS or appointment_type not in AppointmentType:
            raise ControlledIntegrationError("Unsupported demo slot search")
        return [slot for slot in self._slots(appointment_type) if slot.slot_id not in self._booked_slots]

    async def book_appointment(self, request: BookingRequest, idempotency_key: str) -> BookingConfirmation:
        if not idempotency_key.strip():
            raise ControlledIntegrationError("Missing demo booking key")
        previous = self._bookings.get(idempotency_key)
        if previous is not None:
            if request != previous[0]:
                raise ControlledIntegrationError("Demo booking key conflicts with an earlier request")
            return previous[1].model_copy(deep=True)
        slots = [slot for kind in AppointmentType for slot in self._slots(kind)]
        if (
            request.pet_id not in DEMO_PET_IDS or not request.confirmed_by_caller
            or request.slot_id not in {slot.slot_id for slot in slots}
            or request.slot_id in self._booked_slots
        ):
            raise ControlledIntegrationError("Unsupported demo booking request")
        confirmation = BookingConfirmation(
            booking_id=f"demo-booking-{request.slot_id}", slot_id=request.slot_id,
            pet_id=request.pet_id, status="confirmed",
        )
        self._bookings[idempotency_key] = (request.model_copy(deep=True), confirmation)
        self._booked_slots.add(request.slot_id)
        return confirmation.model_copy(deep=True)


class DemoHandoff:
    async def create_handoff(self, request: HandoffRequest) -> HandoffReceipt:
        return HandoffReceipt(
            case_id="demo-handoff-1", conversation_id=request.conversation_id,
            status="queued", received_at=datetime(2026, 11, 3, 12, tzinfo=timezone.utc),
        )


def create_live_run_conversation(
    openai_client: AsyncOpenAI,
    router_model: str,
    specialist_model: str,
    renderer: PromptRenderer,
) -> Callable[[ConversationEvalCase], Awaitable[ConversationRunObservation]]:
    """Run caller turns through real production code; expectations never enter the runtime."""

    async def run_conversation(case: ConversationEvalCase) -> ConversationRunObservation:
        try:
            conversation_id = f"demo-conversation-{uuid4().hex}"
            store = InMemoryConversationStore()
            events: list[WorkflowEvent] = []
            policy_agent = PolicyAgent(
                client=openai_client, model=specialist_model, prompt_renderer=renderer,
                search_policy_handler=create_search_policy_handler(DemoPolicyRetriever()),
            )
            runtime = PromptAgentRuntime(
                conversation_store=store, openai_client=openai_client,
                router_model=router_model, specialist_model=specialist_model,
                prompt_renderer=renderer, legacy_crm_client=DemoCRM(),
                scheduler_client=DemoScheduler(), handoff_client=DemoHandoff(),
                policy_agent=policy_agent,
                idempotency_key_factory=lambda: f"demo-booking-key-{uuid4().hex}",
                workflow_observer=events.append,
            )
            await runtime.start_conversation(conversation_id)
            specialists = []
            tools = []
            for turn_number, caller_message in enumerate(case.caller_turns, start=1):
                await runtime.handle_turn(conversation_id, caller_message)
                turn_events = [event for event in events if event.turn_number == turn_number]
                routes = [event.specialist for event in turn_events if event.event_type == WorkflowEventType.SPECIALIST_SELECTED]
                if len(routes) != 1:
                    # An early terminal state can skip routing for subsequent caller turns.
                    raise LiveConversationEvaluationError("A caller turn did not produce exactly one Router decision")
                specialists.append(routes[0])
                tools.append([event.tool_name for event in turn_events if event.event_type == WorkflowEventType.TOOL_REQUESTED])
            state = store.get(conversation_id)
            return ConversationRunObservation(
                specialists_by_turn=specialists, tools_by_turn=tools,
                final_phase=state.phase, verified_customer=state.verified_customer is not None,
                booking_confirmed=state.booking_confirmation is not None,
                handoff_queued=state.handoff_receipt is not None,
            )
        except Exception:
            raise LiveConversationEvaluationError("The controlled conversation workflow could not complete") from None

    return run_conversation


async def _evaluate(
    cases: list[ConversationEvalCase], api_key: str, router_model: str, specialist_model: str,
) -> dict[str, Any]:
    # Ignore OPENAI_BASE_URL: this evaluator permits OpenAI as its only network service.
    async with AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1") as openai_client:
        runner = create_live_run_conversation(openai_client, router_model, specialist_model, PromptRenderer())
        return await run_conversation_evaluation(cases, runner)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-live", action="store_true", required=True, help="Confirm OpenAI requests and API cost")
    parser.add_argument("--router-model", default=os.getenv("OPENAI_ROUTER_MODEL"))
    parser.add_argument("--specialist-model", default=os.getenv("OPENAI_SPECIALIST_MODEL"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id", action="append", help="Repeat to select several cases")
    selection.add_argument("--all-cases", action="store_true")
    args = parser.parse_args(argv)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        parser.error("OPENAI_API_KEY is required for live evaluation")
    if not args.router_model or not args.router_model.strip():
        parser.error("Set OPENAI_ROUTER_MODEL or --router-model to a nonblank model")
    if not args.specialist_model or not args.specialist_model.strip():
        parser.error("Set OPENAI_SPECIALIST_MODEL or --specialist-model to a nonblank model")
    try:
        cases = load_conversation_cases()
    except (OSError, ValueError):
        parser.error("Unable to load the conversation evaluation dataset")
    requested_ids = set(args.case_id or [])
    if requested_ids - {case.case_id for case in cases}:
        parser.error("Unknown conversation case IDs")
    selected_cases = [case for case in cases if args.all_cases or case.case_id in requested_ids]
    try:
        report = asyncio.run(_evaluate(selected_cases, api_key, args.router_model.strip(), args.specialist_model.strip()))
    except Exception:
        print("Live conversation evaluation failed; no report was produced.", file=sys.stderr)
        return 1
    try:
        attributions = attribute_report_failures(report, eval_type="conversation")
        localization = format_failure_attribution_report(attributions) if attributions else None
    except Exception:
        print("Live conversation failure localization failed; no report was produced.", file=sys.stderr)
        return 1
    try:
        text = format_conversation_report(report)
        if api_key in text:
            raise LiveConversationEvaluationError("Unsafe evaluation report")
    except Exception:
        print("Live conversation evaluation failed; no report was produced.", file=sys.stderr)
        return 1
    print(text)
    if localization is not None:
        print()
        print(localization)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
