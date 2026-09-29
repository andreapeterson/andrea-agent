"""Deterministic, phase-guarded orchestration for one PawLine caller turn."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Protocol

from app.integrations.handoff_client import HandoffClientError
from app.integrations.scheduler_client import (
    IdempotencyConflictError,
    SchedulerRequestError,
    SlotNotFoundError,
    SlotUnavailableError,
)
from app.integrations.turn_interpreter import TurnInterpreter
from app.models.agent_result import AgentTurnResult
from app.models.agent_turn import ExtractedIntakeUpdates, TurnUnderstanding
from app.models.appointment import AppointmentSlot, AppointmentType, BookingRequest
from app.models.conversation import ConversationMessage, ConversationPhase, ConversationRole, ConversationState
from app.models.customer import Customer
from app.models.handoff import HandoffRequest
from app.models.policy_answer import PolicyAnswerResponse
from app.models.routing import RoutingAction, RoutingDecision
from app.services.conversation_store import ConversationNotFoundError, ConversationStore
from app.services.handoff_summary import build_handoff_summary
from app.services.routing import assess_routing


class AgentOrchestrationError(RuntimeError):
    """Base error for deterministic turn orchestration issues."""


class AgentStateError(AgentOrchestrationError):
    """Raised when conversation state or orchestration invariants are invalid."""


class AgentToolError(AgentOrchestrationError):
    """Raised when an integration call fails unexpectedly."""


class LegacyCRMClientProtocol(Protocol):
    async def find_customer_by_phone(self, phone: str) -> Customer | None: ...


class SchedulerClientProtocol(Protocol):
    async def find_slots(self, pet_id: str, appointment_type: AppointmentType) -> list[AppointmentSlot]: ...
    async def book_appointment(self, booking_request: BookingRequest, idempotency_key: str) -> object: ...


class HandoffClientProtocol(Protocol):
    async def create_handoff(self, handoff_request: HandoffRequest) -> object: ...


class PolicyAnswerServiceProtocol(Protocol):
    async def answer(self, question: str) -> PolicyAnswerResponse: ...


class AgentOrchestrator:
    """Process one caller turn with deterministic state transitions.

    This in-memory orchestrator is intentionally process-local. It serializes
    overlapping updates per conversation ID but does not provide cross-process
    coordination or database durability.
    """

    def __init__(
        self,
        conversation_store: ConversationStore,
        turn_interpreter: TurnInterpreter,
        legacy_crm_client: LegacyCRMClientProtocol,
        scheduler_client: SchedulerClientProtocol,
        handoff_client: HandoffClientProtocol,
        policy_service: PolicyAnswerServiceProtocol,
        idempotency_key_factory: Callable[[], str],
    ) -> None:
        self._conversation_store = conversation_store
        self._turn_interpreter = turn_interpreter
        self._legacy_crm_client = legacy_crm_client
        self._scheduler_client = scheduler_client
        self._handoff_client = handoff_client
        self._policy_service = policy_service
        self._idempotency_key_factory = idempotency_key_factory
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        return self._locks.setdefault(conversation_id, asyncio.Lock())

    async def _finalize_turn(
        self,
        state: ConversationState,
        assistant_message: str,
        *,
        policy_answer: PolicyAnswerResponse | None = None,
    ) -> AgentTurnResult:
        trimmed = assistant_message.strip()
        if not trimmed:
            raise AgentStateError("Assistant message cannot be blank.")

        state.messages.append(ConversationMessage(role=ConversationRole.ASSISTANT, content=trimmed))
        saved = self._conversation_store.save(state)
        return AgentTurnResult(state=saved, assistant_message=trimmed, policy_answer=policy_answer)

    @staticmethod
    def _intake_question_for(field_name: str) -> str:
        mapping = {
            "difficulty_breathing": "Is your pet having difficulty breathing?",
            "uncontrolled_bleeding": "Is there uncontrolled bleeding?",
            "collapsed_or_unresponsive": "Is your pet collapsed or unresponsive?",
            "known_toxin_exposure": "Has your pet been exposed to a toxin?",
            "rapidly_worsening": "Is the condition rapidly worsening?",
        }
        return mapping.get(field_name, "Please answer the missing intake question.")

    @staticmethod
    def _normalize_phone(value: str | None) -> str | None:
        if value is None:
            return None
        phone = value.strip()
        return phone or None

    def _resolve_pet(self, state: ConversationState, pet_reference: str | None) -> object | None:
        if state.verified_customer is None:
            return None
        if not pet_reference or not pet_reference.strip():
            return None
        reference = pet_reference.strip()
        exact_id = next((pet for pet in state.verified_customer.pets if pet.pet_id == reference), None)
        if exact_id is not None:
            return exact_id
        lowered = reference.casefold()
        matches = [pet for pet in state.verified_customer.pets if pet.name.casefold() == lowered]
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_slot(self, state: ConversationState, selection: str | None) -> AppointmentSlot | None:
        if selection is None or not selection.strip():
            return None
        candidate = selection.strip()
        for slot in state.offered_slots:
            if slot.slot_id == candidate:
                return slot
        ordinal_map = {
            "first": 0,
            "second": 1,
            "third": 2,
            "fourth": 3,
            "fifth": 4,
        }
        index = ordinal_map.get(candidate.casefold())
        if index is not None and 0 <= index < len(state.offered_slots):
            return state.offered_slots[index]
        return None

    def _merge_intake_updates(self, state: ConversationState, updates: ExtractedIntakeUpdates) -> None:
        for field_name in [
            "difficulty_breathing",
            "uncontrolled_bleeding",
            "collapsed_or_unresponsive",
            "known_toxin_exposure",
            "rapidly_worsening",
        ]:
            value = getattr(updates, field_name)
            if value is not None:
                setattr(state.intake_answers, field_name, value)

    async def _create_handoff(self, state: ConversationState) -> AgentTurnResult:
        if state.handoff_receipt is not None:
            return await self._finalize_turn(state, "A human handoff was already requested for this conversation.")
        if state.verified_customer is None or state.selected_pet_id is None or not state.original_concern:
            raise AgentStateError("Urgent handoff requires a verified customer, selected pet, and concern.")

        summary = build_handoff_summary(
            original_concern=state.original_concern,
            intake_answers=state.intake_answers,
            routing_decision=state.routing_decision,
        )
        request = HandoffRequest(
            conversation_id=state.conversation_id,
            verified_customer_id=state.verified_customer.customer_id,
            selected_pet_id=state.selected_pet_id,
            original_concern=state.original_concern,
            intake_answers=state.intake_answers,
            routing_decision=state.routing_decision,
            summary=summary,
        )
        try:
            receipt = await self._handoff_client.create_handoff(request)
        except (HandoffClientError, RuntimeError) as exc:
            raise AgentToolError("Handoff creation failed.") from exc

        state.handoff_receipt = receipt
        state.phase = ConversationPhase.HANDOFF_COMPLETE
        return await self._finalize_turn(state, "A human handoff was requested.")

    async def _search_slots(self, state: ConversationState) -> list[AppointmentSlot]:
        if state.selected_pet_id is None or state.appointment_type is None:
            raise AgentStateError("Appointment search requires a selected pet and appointment type.")
        try:
            slots = await self._scheduler_client.find_slots(
                pet_id=state.selected_pet_id,
                appointment_type=state.appointment_type,
            )
        except (SchedulerRequestError, RuntimeError) as exc:
            raise AgentToolError("Scheduler slot search failed.") from exc
        state.offered_slots = slots
        return slots

    def _format_slots(self, slots: list[AppointmentSlot]) -> str:
        if not slots:
            return "No appointment slots are currently available."
        parts = [f"{index + 1}. {slot.slot_id} at {slot.starts_at.strftime('%Y-%m-%d %H:%M')}" for index, slot in enumerate(slots)]
        return "Please choose one of these options: " + "; ".join(parts)

    async def _book_selected_slot(self, state: ConversationState) -> AgentTurnResult:
        if state.booking_confirmation is not None:
            return await self._finalize_turn(state, "The booking was already confirmed.")
        if state.selected_slot_id is None:
            raise AgentStateError("Booking requires a selected slot.")
        if state.selected_pet_id is None or state.verified_customer is None:
            raise AgentStateError("Booking requires a verified customer and selected pet.")

        if state.booking_idempotency_key is None:
            state.booking_idempotency_key = self._idempotency_key_factory()
            self._conversation_store.save(state)

        booking_request = BookingRequest(
            slot_id=state.selected_slot_id,
            pet_id=state.selected_pet_id,
            confirmed_by_caller=True,
        )

        try:
            confirmation = await self._scheduler_client.book_appointment(
                booking_request=booking_request,
                idempotency_key=state.booking_idempotency_key,
            )
        except SlotUnavailableError as exc:
            state.selected_slot_id = None
            state.booking_idempotency_key = None
            refreshed = await self._scheduler_client.find_slots(
                pet_id=state.selected_pet_id,
                appointment_type=state.appointment_type,
            )
            state.offered_slots = refreshed
            if refreshed:
                state.phase = ConversationPhase.SELECTING_APPOINTMENT
                return await self._finalize_turn(state, "The selected slot is no longer available. Please choose another option.")
            state.phase = ConversationPhase.COMPLETED
            return await self._finalize_turn(state, "The selected slot disappeared and no alternatives are currently available.")
        except SlotNotFoundError as exc:
            state.selected_slot_id = None
            state.booking_idempotency_key = None
            refreshed = await self._scheduler_client.find_slots(
                pet_id=state.selected_pet_id,
                appointment_type=state.appointment_type,
            )
            state.offered_slots = refreshed
            if refreshed:
                state.phase = ConversationPhase.SELECTING_APPOINTMENT
                return await self._finalize_turn(state, "That slot is no longer available. Please choose another option.")
            state.phase = ConversationPhase.COMPLETED
            return await self._finalize_turn(state, "The selected slot was unavailable and no alternatives are available.")
        except IdempotencyConflictError as exc:
            raise AgentStateError("Booking idempotency key conflict indicates an internal invariant problem.") from exc
        except SchedulerRequestError as exc:
            state.phase = ConversationPhase.CONFIRMING_BOOKING
            self._conversation_store.save(state)
            raise AgentToolError("Scheduler booking request failed.") from exc
        except Exception as exc:
            raise AgentToolError("Unexpected scheduler booking failure.") from exc

        state.booking_confirmation = confirmation
        state.phase = ConversationPhase.BOOKING_COMPLETE
        return await self._finalize_turn(state, "Your appointment was confirmed.")

    async def start_conversation(self, conversation_id: str) -> AgentTurnResult:
        if conversation_id is None or not isinstance(conversation_id, str):
            raise AgentStateError("conversation_id must be a non-empty string.")
        normalized_id = conversation_id.strip()
        if not normalized_id:
            raise AgentStateError("conversation_id cannot be blank.")

        lock = self._get_lock(normalized_id)
        async with lock:
            state = self._conversation_store.create(normalized_id)
            state.phase = ConversationPhase.VERIFYING_CUSTOMER
            return await self._finalize_turn(state, "Please provide the caller's phone number.")

    async def _answer_policy_question(
        self,
        state: ConversationState,
        question: str,
    ) -> AgentTurnResult:
        try:
            policy_answer = await self._policy_service.answer(question)
        except Exception as exc:
            raise AgentToolError("Policy question handling failed.") from exc

        answer_text = policy_answer.answer.strip()
        if not answer_text:
            answer_text = "I’m not able to answer that policy question from the current evidence."

        workflow_message = ""
        if state.phase == ConversationPhase.VERIFYING_CUSTOMER:
            workflow_message = "Please provide the caller's phone number."
        elif state.phase == ConversationPhase.SELECTING_PET:
            names = ", ".join(pet.name for pet in state.verified_customer.pets) if state.verified_customer else ""
            workflow_message = f"Which pet are you calling about? {names}" if names else "Which pet are you calling about?"
        elif state.phase == ConversationPhase.COLLECTING_CONCERN:
            workflow_message = "Please tell me the pet's concern."
        elif state.phase == ConversationPhase.COLLECTING_INTAKE:
            workflow_message = "Please answer the remaining intake questions."
        elif state.phase == ConversationPhase.SELECTING_APPOINTMENT:
            workflow_message = self._format_slots(state.offered_slots)
        elif state.phase == ConversationPhase.CONFIRMING_BOOKING:
            workflow_message = "Please confirm the appointment by saying yes or no."

        message = answer_text
        if workflow_message:
            message = f"{answer_text} {workflow_message}"

        return await self._finalize_turn(state, message, policy_answer=policy_answer)

    async def handle_turn(self, conversation_id: str, user_message: str) -> AgentTurnResult:
        if conversation_id is None or not isinstance(conversation_id, str):
            raise AgentStateError("conversation_id must be a non-empty string.")
        normalized_id = conversation_id.strip()
        if not normalized_id:
            raise AgentStateError("conversation_id cannot be blank.")

        if user_message is None or not isinstance(user_message, str):
            raise AgentStateError("user_message must be a non-empty string.")
        cleaned_message = user_message.strip()
        if not cleaned_message:
            raise AgentStateError("user_message cannot be blank.")

        lock = self._get_lock(normalized_id)
        async with lock:
            state = self._conversation_store.get(normalized_id)

            if state.phase in {
                ConversationPhase.HANDOFF_COMPLETE,
                ConversationPhase.BOOKING_COMPLETE,
                ConversationPhase.COMPLETED,
            }:
                return await self._finalize_turn(state, "This conversation is already complete.")

            if state.phase == ConversationPhase.STARTED:
                state.phase = ConversationPhase.VERIFYING_CUSTOMER

            interpretation = await self._turn_interpreter.interpret(state, cleaned_message)
            state.messages.append(ConversationMessage(role=ConversationRole.USER, content=cleaned_message))
            state.turn_count += 1

            if state.phase == ConversationPhase.VERIFYING_CUSTOMER and interpretation.policy_question and interpretation.policy_question.strip():
                return await self._answer_policy_question(state, interpretation.policy_question)

            if state.phase == ConversationPhase.VERIFYING_CUSTOMER:
                phone = self._normalize_phone(interpretation.phone_number)
                if phone is None:
                    return await self._finalize_turn(state, "Please provide the caller's phone number.")
                try:
                    customer = await self._legacy_crm_client.find_customer_by_phone(phone)
                except Exception as exc:
                    raise AgentToolError("Customer lookup failed.") from exc
                if customer is None:
                    return await self._finalize_turn(state, "I could not find that customer. Please try again.")
                state.verified_customer = customer
                if not customer.pets:
                    state.phase = ConversationPhase.COMPLETED
                    return await self._finalize_turn(state, "I could not find any pets on that account.")
                if len(customer.pets) == 1:
                    state.selected_pet_id = customer.pets[0].pet_id
                    state.phase = ConversationPhase.COLLECTING_CONCERN
                    return await self._finalize_turn(state, "I found the pet. Please tell me what happened.")
                resolved = self._resolve_pet(state, interpretation.pet_reference)
                if resolved is None:
                    state.phase = ConversationPhase.SELECTING_PET
                    pet_names = ", ".join(pet.name for pet in customer.pets)
                    return await self._finalize_turn(state, f"Which pet are you calling about? {pet_names}")
                state.selected_pet_id = resolved.pet_id
                state.phase = ConversationPhase.COLLECTING_CONCERN
                return await self._finalize_turn(state, "I found the pet. Please tell me what happened.")

            if state.phase == ConversationPhase.SELECTING_PET:
                resolved = self._resolve_pet(state, interpretation.pet_reference)
                if resolved is None:
                    names = ", ".join(pet.name for pet in state.verified_customer.pets)
                    return await self._finalize_turn(state, f"Which pet are you calling about? {names}")
                state.selected_pet_id = resolved.pet_id
                state.phase = ConversationPhase.COLLECTING_CONCERN
                return await self._finalize_turn(state, "I found the pet. Please tell me what happened.")

            if state.phase == ConversationPhase.COLLECTING_CONCERN:
                concern = (interpretation.original_concern or "").strip()
                if not concern:
                    return await self._finalize_turn(state, "Please tell me the pet's concern.")
                state.original_concern = concern
                state.phase = ConversationPhase.COLLECTING_INTAKE
                return await self._finalize_turn(state, "I have the concern. I need a few quick intake questions.")

            if state.phase == ConversationPhase.COLLECTING_INTAKE:
                self._merge_intake_updates(state, interpretation.intake_updates)
                decision = assess_routing(state.intake_answers)
                state.routing_decision = decision

                if decision.next_action == RoutingAction.CREATE_HANDOFF:
                    return await self._create_handoff(state)

                if decision.next_action == RoutingAction.ASK_INTAKE_QUESTION:
                    missing_field = decision.missing_fields[0] if decision.missing_fields else None
                    if missing_field is None:
                        return await self._finalize_turn(state, "I need a little more detail about the pet's condition.")
                    return await self._finalize_turn(state, self._intake_question_for(missing_field))

                if decision.next_action in {RoutingAction.SEARCH_SAME_DAY_APPOINTMENT, RoutingAction.SEARCH_ROUTINE_APPOINTMENT}:
                    state.appointment_type = (
                        AppointmentType.SAME_DAY
                        if decision.next_action == RoutingAction.SEARCH_SAME_DAY_APPOINTMENT
                        else AppointmentType.ROUTINE
                    )
                    slots = await self._search_slots(state)
                    if not slots:
                        state.phase = ConversationPhase.COMPLETED
                        return await self._finalize_turn(state, "No matching appointments are currently available.")
                    state.offered_slots = slots
                    state.phase = ConversationPhase.SELECTING_APPOINTMENT
                    return await self._finalize_turn(state, self._format_slots(slots))

            if state.phase == ConversationPhase.SELECTING_APPOINTMENT:
                selected_slot = self._resolve_slot(state, interpretation.appointment_selection)
                if selected_slot is None:
                    return await self._finalize_turn(state, self._format_slots(state.offered_slots))
                state.selected_slot_id = selected_slot.slot_id
                state.phase = ConversationPhase.CONFIRMING_BOOKING
                return await self._finalize_turn(state, "Please confirm the appointment by saying yes or no.")

            if state.phase == ConversationPhase.CONFIRMING_BOOKING:
                if interpretation.booking_confirmed is None:
                    return await self._finalize_turn(state, "Please confirm the appointment by saying yes or no.")
                if interpretation.booking_confirmed is False:
                    state.selected_slot_id = None
                    state.booking_idempotency_key = None
                    state.phase = ConversationPhase.SELECTING_APPOINTMENT
                    return await self._finalize_turn(state, self._format_slots(state.offered_slots))
                if interpretation.booking_confirmed is True:
                    if state.selected_slot_id is None:
                        raise AgentStateError("Selected slot is required before booking.")
                    return await self._book_selected_slot(state)

            if interpretation.policy_question and interpretation.policy_question.strip():
                return await self._answer_policy_question(state, interpretation.policy_question)

            if state.phase == ConversationPhase.STARTED:
                state.phase = ConversationPhase.VERIFYING_CUSTOMER
                return await self._finalize_turn(state, "Please provide the caller's phone number.")

            return await self._finalize_turn(state, "I’m ready to continue.")
