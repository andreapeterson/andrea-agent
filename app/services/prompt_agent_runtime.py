"""Coordinate one complete caller turn using the existing router and specialists."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from openai import AsyncOpenAI

from app.agents.human_handoff_agent import HumanHandoffAgent
from app.agents.intake_agent import IntakeAgent
from app.agents.lookup_customer_tool import create_lookup_customer_handler
from app.agents.policy_agent import PolicyAgent
from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.record_intake_tool import create_record_intake_handler
from app.agents.request_handoff_tool import create_request_handoff_handler
from app.agents.router_agent import OpenAIRouter
from app.agents.router_handoff import route_and_respond
from app.agents.scheduling_agent import SchedulingAgent
from app.agents.scheduling_tools import (
    create_book_selected_appointment_handler,
    create_find_appointment_slots_handler,
    create_select_appointment_slot_handler,
)
from app.agents.verification_agent import VerificationAgent
from app.integrations import HandoffClient, LegacyCRMClient, SchedulerClient
from app.models.agent_result import AgentTurnResult
from app.models.conversation import (
    ConversationMessage,
    ConversationPhase,
    ConversationRole,
    ConversationState,
)
from app.services.conversation_store import ConversationStore


class PromptAgentRuntime:
    """Own one full caller turn, from loading state to saving the specialist's reply."""

    def __init__(
        self,
        *,
        conversation_store: ConversationStore,
        openai_client: AsyncOpenAI,
        router_model: str,
        specialist_model: str,
        prompt_renderer: PromptRenderer,
        legacy_crm_client: LegacyCRMClient,
        scheduler_client: SchedulerClient,
        handoff_client: HandoffClient,
        policy_agent: PolicyAgent | None,
        idempotency_key_factory: Callable[[], str],
    ) -> None:
        self._conversation_store = conversation_store
        self._openai_client = openai_client
        self._router = OpenAIRouter(openai_client, model=router_model)
        self._specialist_model = specialist_model
        self._prompt_renderer = prompt_renderer
        self._legacy_crm_client = legacy_crm_client
        self._scheduler_client = scheduler_client
        self._handoff_client = handoff_client
        self._policy_agent = policy_agent
        self._idempotency_key_factory = idempotency_key_factory
        self._locks: dict[str, asyncio.Lock] = {}

    def _build_prompt_context(self, state: ConversationState) -> PromptContext: # everything Pawline knows to info approved for prompt
        """Copy only allowlisted conversation facts, never the full customer or state."""
        customer = state.verified_customer
        selected_pet = next(
            (pet.name for pet in customer.pets if pet.pet_id == state.selected_pet_id),
            None,
        ) if customer is not None else None
        selected_slot = next(
            (slot for slot in state.offered_slots if slot.slot_id == state.selected_slot_id),
            None,
        )
        # Keep a snapshot: later tool mutations must not change the pre-turn context.
        return PromptContext(
            active_agent="router",
            customer_verified=customer is not None,
            customer_first_name=customer.first_name if customer is not None else None,
            known_pets=[pet.name for pet in customer.pets] if customer is not None else [],
            selected_pet=selected_pet,
            original_concern=state.original_concern,
            intake_answers=state.intake_answers,
            missing_intake_answers=[
                field for field, value in state.intake_answers.model_dump().items()
                if value is None
            ],
            routing_result=state.routing_decision,
            offered_slots=state.offered_slots,
            selected_slot=selected_slot,
            booking_confirmed=state.booking_confirmation is not None,
            recent_messages=state.messages[-6:], #last 6 messages for context
        ).model_copy(deep=True) #create a deep copy to prevent accidental mutation of the original state, describes state before the turn, so that the model can make decisions based on the state before the turn

    async def start_conversation(self, conversation_id: str) -> AgentTurnResult:
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation_id must be a nonblank string.")
        conversation_id = conversation_id.strip()
        lock = self._locks.setdefault(conversation_id, asyncio.Lock()) #if lock doesnt exist create new, or return existing 
        async with lock: #only one turn can be processed at a time for a given conversation_id, preventing race conditions and ensuring that the conversation state is updated in a consistent manner
            state = self._conversation_store.create(conversation_id)
            state.phase = ConversationPhase.VERIFYING_CUSTOMER
            greeting = "Hi, I'm PawLine. How may I help you?"
            state.messages.append(ConversationMessage(role=ConversationRole.ASSISTANT, content=greeting)) #save greeting message to conversation state
            saved = self._conversation_store.save(state)
            return AgentTurnResult(state=saved, assistant_message=greeting) #return state and greeting

    async def handle_turn(self, conversation_id: str, user_message: str) -> AgentTurnResult: #turn = 1 caller message and 1 pawline response
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation_id must be a nonblank string.")
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("User message cannot be blank.")
        conversation_id = conversation_id.strip()
        user_message = user_message.strip()

        # The lock protects this conversation from overlapping loads, tool updates, and saves.
        lock = self._locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            state = self._conversation_store.get(conversation_id)
            if state.phase in {
                ConversationPhase.BOOKING_COMPLETE,
                ConversationPhase.HANDOFF_COMPLETE,
                ConversationPhase.COMPLETED,
            }:
                message = next(
                    (message.content for message in reversed(state.messages) if message.role == ConversationRole.ASSISTANT),
                    "This conversation is already complete.", #return last assistant message if available, otherwise default message
                )
                return AgentTurnResult(state=state, assistant_message=message)

            context = self._build_prompt_context(state)
            # All handlers share this state; service-backed tools also receive their service client.
            lookup_handler = create_lookup_customer_handler(
                self._legacy_crm_client,
                conversation_state=state,
                conversation_store=self._conversation_store,
            )
            intake_handler = create_record_intake_handler(
                conversation_state=state,
                conversation_store=self._conversation_store,
            )
            handoff_handler = create_request_handoff_handler(
                self._handoff_client,
                conversation_state=state,
                conversation_store=self._conversation_store,
            )
            find_slots_handler = create_find_appointment_slots_handler(
                self._scheduler_client,
                conversation_state=state,
                conversation_store=self._conversation_store,
            )
            select_slot_handler = create_select_appointment_slot_handler(
                conversation_state=state,
                conversation_store=self._conversation_store,
            )
            booking_handler = create_book_selected_appointment_handler(
                self._scheduler_client,
                conversation_state=state,
                conversation_store=self._conversation_store,
                idempotency_key_factory=self._idempotency_key_factory,
            )

            # Agents configure model calls. Tool definitions go to the model; handlers run in Python.
            handle = route_and_respond( #configured function that knows which router and specialists to call based on the current conversation state and user message
                router=self._router,
                renderer=self._prompt_renderer,
                verification_agent=VerificationAgent(
                    client=self._openai_client,
                    model=self._specialist_model,
                    prompt_renderer=self._prompt_renderer,
                    lookup_customer_handler=lookup_handler,
                ),
                intake_agent=IntakeAgent(
                    client=self._openai_client,
                    model=self._specialist_model,
                    prompt_renderer=self._prompt_renderer,
                    record_intake_handler=intake_handler,
                    request_handoff_handler=handoff_handler,
                ),
                human_handoff_agent=HumanHandoffAgent(
                    client=self._openai_client,
                    model=self._specialist_model,
                    prompt_renderer=self._prompt_renderer,
                    request_handoff_handler=handoff_handler,
                ),
                scheduling_agent=SchedulingAgent(
                    client=self._openai_client,
                    model=self._specialist_model,
                    prompt_renderer=self._prompt_renderer,
                    find_appointment_slots_handler=find_slots_handler,
                    select_appointment_slot_handler=select_slot_handler,
                    book_selected_appointment_handler=booking_handler,
                ),
                policy_agent=self._policy_agent,
            )
            # On an error, do not overwrite important state already saved by a tool.
            assistant_message = await handle(context=context, user_message=user_message) #actually call the router and specialist agents to get a response based on the user message and current context
            state.messages.extend([
                ConversationMessage(role=ConversationRole.USER, content=user_message),
                ConversationMessage(role=ConversationRole.ASSISTANT, content=assistant_message),
            ])
            state.turn_count += 1
            saved = self._conversation_store.save(state)
            return AgentTurnResult(state=saved, assistant_message=assistant_message)


__all__ = ["PromptAgentRuntime"]
