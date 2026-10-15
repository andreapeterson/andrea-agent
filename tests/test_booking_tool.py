"""Verify explicit booking, persisted retries, and recovery without network requests."""

from __future__ import annotations

import json
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from app.agents import PromptContext, PromptRenderer, SchedulingAgent
from app.agents.scheduling_tools import (
    BOOK_SELECTED_APPOINTMENT_TOOL,
    FIND_APPOINTMENT_SLOTS_TOOL,
    SELECT_APPOINTMENT_SLOT_TOOL,
    create_book_selected_appointment_handler,
)
from app.integrations.scheduler_client import (
    CallerConfirmationRequiredError,
    IdempotencyConflictError,
    SchedulerClient,
    SchedulerRequestError,
    SchedulerResponseError,
    SlotNotFoundError,
    SlotUnavailableError,
)
from app.models.appointment import AppointmentType, BookingConfirmation, BookingStatus
from app.models.conversation import ConversationPhase
from mock_services import scheduling_api
from tests.test_scheduling_agent import FakeResponses, fake_client, output, prepared, slot


def confirmation(slot_id: str = "slot-1", pet_id: str = "pet-1") -> BookingConfirmation:
    return BookingConfirmation(booking_id="booking-1", slot_id=slot_id, pet_id=pet_id, status=BookingStatus.CONFIRMED)


class FakeBookingScheduler:
    def __init__(self, outcomes: list[object] | None = None, refresh: object | None = None) -> None:
        self.outcomes = [confirmation()] if outcomes is None else list(outcomes)
        self.refresh = [] if refresh is None else refresh
        self.book_calls = []
        self.find_calls = []
        self.store = None

    async def book_appointment(self, request, key):
        # Every external send must already have its trusted request and key saved.
        saved = self.store.get("conv-scheduling")
        assert saved.booking_idempotency_key == key
        assert saved.selected_slot_id == request.slot_id
        assert saved.selected_pet_id == request.pet_id
        self.book_calls.append((request.model_copy(deep=True), key))
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def find_slots(self, pet_id, appointment_type):
        self.find_calls.append((pet_id, appointment_type))
        if isinstance(self.refresh, Exception):
            raise self.refresh
        return self.refresh


def setup_booking(client, *, factory=None, chosen=None):
    state, store, find, select = prepared(client)
    state.selected_pet_id = "pet-1"
    state.appointment_type = AppointmentType.ROUTINE
    state.offered_slots = [slot() if chosen is None else chosen]
    state.selected_slot_id = state.offered_slots[0].slot_id
    state.phase = ConversationPhase.CONFIRMING_BOOKING
    store.save(state)
    if isinstance(client, FakeBookingScheduler):
        client.store = store
    factory = Mock(return_value="private-key-1") if factory is None else factory
    book = create_book_selected_appointment_handler(
        client, conversation_state=state, conversation_store=store, idempotency_key_factory=factory,
    )
    return state, store, find, select, book, factory


def test_booking_schema_accepts_only_a_required_strict_confirmation_boolean() -> None:
    schema = BOOK_SELECTED_APPOINTMENT_TOOL["parameters"]
    assert BOOK_SELECTED_APPOINTMENT_TOOL["strict"] is True
    assert BOOK_SELECTED_APPOINTMENT_TOOL["name"] == "book_selected_appointment"
    assert schema["required"] == ["confirmed_by_caller"]
    assert schema["additionalProperties"] is False
    assert list(schema["properties"]) == ["confirmed_by_caller"]
    assert schema["properties"]["confirmed_by_caller"]["type"] == "boolean"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {}, {"confirmed_by_caller": "true"}, {"confirmed_by_caller": 1}, {"confirmed_by_caller": None},
    *({"confirmed_by_caller": True, field: "injected"} for field in [
        "slot_id", "pet_id", "appointment_type", "booking_id", "idempotency_key", "customer_id", "verified_customer",
    ]),
])
async def test_model_cannot_inject_booking_identity_or_confirmation(payload) -> None:
    client = FakeBookingScheduler()
    state, store, _, _, book, factory = setup_booking(client)
    before = store.get(state.conversation_id)
    with pytest.raises(ValidationError):
        await book(payload)
    assert client.book_calls == []
    assert state == before
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_decline_before_any_booking_request_clears_selection_without_scheduler_calls() -> None:
    client = FakeBookingScheduler()
    state, store, _, _, book, factory = setup_booking(client)
    offered = list(state.offered_slots)

    result = await book({"confirmed_by_caller": False})

    assert result == {"status": "booking_declined", "slots": [item.model_dump(mode="json") for item in offered]}
    assert client.book_calls == client.find_calls == []
    factory.assert_not_called()
    saved = store.get(state.conversation_id)
    assert saved.phase == ConversationPhase.SELECTING_APPOINTMENT
    assert saved.selected_slot_id is None
    assert saved.booking_idempotency_key is None
    assert saved.offered_slots == offered


@pytest.mark.asyncio
async def test_decline_with_saved_key_requires_recovery_and_leaves_state_unchanged() -> None:
    client = FakeBookingScheduler()
    state, store, _, _, book, factory = setup_booking(client)
    state.booking_idempotency_key = "earlier-uncertain-key"
    store.save(state)
    before = state.model_copy(deep=True)
    store.save = Mock(wraps=store.save)

    result = await book({"confirmed_by_caller": False})

    assert result == {"status": "booking_recovery_required", "booking_outcome_uncertain": True}
    assert state == before
    assert store.get(state.conversation_id) == before
    store.save.assert_not_called()
    assert client.book_calls == client.find_calls == []
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_uncertain_attempt_cannot_be_declined_or_changed_to_another_offered_slot() -> None:
    client = FakeBookingScheduler([SchedulerRequestError("Lost response"), confirmation()])
    state, store, find, select, book, factory = setup_booking(client)
    state.offered_slots.append(slot("slot-2"))
    assert await book({"confirmed_by_caller": True}) == {"status": "scheduler_unavailable", "retry_allowed": True}
    before = store.get(state.conversation_id)
    store.save = Mock(wraps=store.save)

    assert await book({"confirmed_by_caller": False}) == {
        "status": "booking_recovery_required", "booking_outcome_uncertain": True,
    }
    assert await select({"slot_id": "slot-2"}) == {"status": "booking_recovery_required"}
    assert await find({"pet_reference": None}) == {"status": "booking_recovery_required"}
    assert state == before
    assert store.get(state.conversation_id) == before
    store.save.assert_not_called()
    assert len(client.book_calls) == 1
    assert client.find_calls == []

    assert (await book({"confirmed_by_caller": True}))["status"] == "booking_confirmed"
    assert [request.slot_id for request, _ in client.book_calls] == ["slot-1", "slot-1"]
    assert [key for _, key in client.book_calls] == ["private-key-1", "private-key-1"]
    factory.assert_called_once_with()


@pytest.mark.asyncio
async def test_explicit_confirmation_persists_key_before_send_and_saves_success() -> None:
    client = FakeBookingScheduler()
    state, store, _, _, book, factory = setup_booking(client)
    result = await book({"confirmed_by_caller": True})

    request, key = client.book_calls[0]
    assert request.model_dump() == {"slot_id": "slot-1", "pet_id": "pet-1", "confirmed_by_caller": True}
    assert key == "private-key-1"
    factory.assert_called_once_with()
    assert result == {"status": "booking_confirmed", "confirmation": confirmation().model_dump(mode="json")}
    assert key not in json.dumps(result)
    saved = store.get(state.conversation_id)
    assert saved.booking_confirmation == confirmation()
    assert saved.phase == ConversationPhase.BOOKING_COMPLETE
    assert saved.booking_idempotency_key == key


@pytest.mark.asyncio
async def test_store_failure_prevents_post_and_retry_persists_same_key_first() -> None:
    client = FakeBookingScheduler()
    state, store, _, _, book, factory = setup_booking(client)
    real_save = store.save
    store.save = Mock(side_effect=RuntimeError("Store temporarily unavailable"))
    with pytest.raises(RuntimeError, match="Store temporarily unavailable"):
        await book({"confirmed_by_caller": True})
    assert client.book_calls == []
    assert store.get(state.conversation_id).booking_idempotency_key is None
    store.save = real_save

    assert (await book({"confirmed_by_caller": True}))["status"] == "booking_confirmed"
    assert client.book_calls[0][1] == "private-key-1"
    factory.assert_called_once_with()


@pytest.mark.asyncio
async def test_uncertain_request_failure_and_reloaded_retry_reuse_same_key() -> None:
    client = FakeBookingScheduler([SchedulerRequestError("private transport exception"), confirmation()])
    state, store, _, _, book, factory = setup_booking(client)
    result = await book({"confirmed_by_caller": True})

    assert result == {"status": "scheduler_unavailable", "retry_allowed": True}
    saved = store.get(state.conversation_id)
    assert saved.phase == ConversationPhase.CONFIRMING_BOOKING
    assert saved.selected_slot_id == "slot-1"
    assert saved.booking_idempotency_key == "private-key-1"
    assert saved.booking_confirmation is None
    retry_factory = Mock(side_effect=AssertionError("Never replace an uncertain attempt's key"))
    retry = create_book_selected_appointment_handler(client, conversation_state=saved, conversation_store=store, idempotency_key_factory=retry_factory)
    assert (await retry({"confirmed_by_caller": True}))["status"] == "booking_confirmed"
    assert [key for _, key in client.book_calls] == ["private-key-1", "private-key-1"]
    assert client.book_calls[0][0] == client.book_calls[1][0]
    factory.assert_called_once_with()
    retry_factory.assert_not_called()


@pytest.mark.asyncio
async def test_saved_confirmation_prevents_duplicate_api_calls_even_after_reloading_state() -> None:
    client = FakeBookingScheduler()
    state, store, _, _, book, factory = setup_booking(client)
    first = await book({"confirmed_by_caller": True})
    reloaded = store.get(state.conversation_id)
    again = create_book_selected_appointment_handler(client, conversation_state=reloaded, conversation_store=store, idempotency_key_factory=factory)
    assert await again({"confirmed_by_caller": True}) == first
    assert await book({"confirmed_by_caller": False}) == first
    assert len(client.book_calls) == 1
    factory.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation, status", [
    ("unverified", "verification_required"), ("urgent", "handoff_required"), ("missing_intake", "intake_required"),
    ("handoff_complete", "handoff_complete"), ("booking_complete", "booking_complete"), ("completed", "conversation_complete"),
    ("foreign_pet", "pet_clarification_required"), ("no_pet", "pet_clarification_required"),
    ("no_selection", "selection_required"), ("foreign_slot", "slot_not_offered"),
    ("wrong_type", "search_required"), ("not_confirming", "confirmation_required"),
])
async def test_booking_rechecks_trusted_state_before_generating_key_or_contacting_scheduler(mutation, status) -> None:
    client = FakeBookingScheduler()
    state, store, _, _, book, factory = setup_booking(client)
    if mutation == "unverified":
        state.verified_customer_id = state.verified_customer.customer_id
        state.verified_customer = None
    elif mutation == "urgent":
        state.intake_answers.difficulty_breathing = True
    elif mutation == "missing_intake":
        state.intake_answers.rapidly_worsening = None
    elif mutation in {"handoff_complete", "booking_complete", "completed"}:
        state.phase = ConversationPhase(mutation)
    elif mutation == "foreign_pet":
        state.selected_pet_id = "unowned"
    elif mutation == "no_pet":
        state.selected_pet_id = None
    elif mutation == "no_selection":
        state.selected_slot_id = None
    elif mutation == "foreign_slot":
        state.selected_slot_id = "not-offered"
    elif mutation == "wrong_type":
        state.offered_slots = [slot(appointment_type=AppointmentType.SAME_DAY)]
    elif mutation == "not_confirming":
        state.phase = ConversationPhase.SELECTING_APPOINTMENT
    before = state.model_copy(deep=True)
    result = await book({"confirmed_by_caller": True})
    assert result["status"] == status
    assert client.book_calls == client.find_calls == []
    assert state == before
    factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome, status", [
    (confirmation(slot_id="wrong-slot"), "invalid_scheduler_response"),
    (confirmation(pet_id="wrong-pet"), "invalid_scheduler_response"),
    ({"status": "confirmed"}, "invalid_scheduler_response"),
    ({**confirmation().model_dump(mode="json"), "status": "pending"}, "invalid_scheduler_response"),
    ({**confirmation().model_dump(mode="json"), "booking_id": " "}, "invalid_scheduler_response"),
    (SchedulerResponseError("private malformed response"), "invalid_scheduler_response"),
    (IdempotencyConflictError("private invariant error"), "idempotency_conflict"),
    (CallerConfirmationRequiredError("private invalid request"), "invalid_booking_request"),
])
async def test_invalid_or_conflicting_response_keeps_uncertain_attempt_and_never_claims_success(outcome, status) -> None:
    client = FakeBookingScheduler([outcome])
    state, store, _, _, book, factory = setup_booking(client)
    result = await book({"confirmed_by_caller": True})
    assert result == {"status": status}
    assert "private" not in json.dumps(result)
    saved = store.get(state.conversation_id)
    assert saved.booking_confirmation is None
    assert saved.phase == ConversationPhase.CONFIRMING_BOOKING
    assert saved.selected_slot_id == "slot-1"
    assert saved.booking_idempotency_key == "private-key-1"
    assert len(client.book_calls) == 1
    assert client.find_calls == []
    factory.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [SlotUnavailableError("Taken"), SlotNotFoundError("Disappeared")])
async def test_disappeared_slot_refreshes_alternatives_and_replacement_receives_new_key(error) -> None:
    factory = Mock(side_effect=["failed-slot-key", "replacement-slot-key"])
    client = FakeBookingScheduler([error, confirmation(slot_id="slot-2")], refresh=[slot("slot-2")])
    state, store, _, select, book, _ = setup_booking(client, factory=factory)
    result = await book({"confirmed_by_caller": True})
    assert result == {"status": "slot_unavailable", "selected_pet": {"pet_id": "pet-1", "name": "Milo"}, "appointment_type": "routine", "slots": [slot("slot-2").model_dump(mode="json")]}
    assert client.find_calls == [("pet-1", AppointmentType.ROUTINE)]
    saved = store.get(state.conversation_id)
    assert saved.selected_slot_id is None
    assert saved.booking_idempotency_key is None
    assert saved.offered_slots == [slot("slot-2")]
    assert saved.phase == ConversationPhase.SELECTING_APPOINTMENT
    await select({"slot_id": "slot-2"})
    assert (await book({"confirmed_by_caller": True}))["status"] == "booking_confirmed"
    assert [key for _, key in client.book_calls] == ["failed-slot-key", "replacement-slot-key"]
    assert factory.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("refresh", [[], SchedulerRequestError("Refresh unavailable"), SchedulerResponseError("Bad refresh"), [{"broken": "slot"}]])
async def test_disappeared_slot_preserves_valid_unbooked_state_even_without_alternatives(refresh) -> None:
    client = FakeBookingScheduler([SlotUnavailableError("Taken")], refresh=refresh)
    state, store, _, _, book, _ = setup_booking(client)
    result = await book({"confirmed_by_caller": True})
    assert result["status"] == ("slot_unavailable" if refresh == [] else "search_required")
    assert result["slots"] == []
    saved = store.get(state.conversation_id)
    assert saved.selected_slot_id is None
    assert saved.booking_idempotency_key is None
    assert saved.booking_confirmation is None
    assert saved.offered_slots == []
    assert saved.phase == ConversationPhase.SELECTING_APPOINTMENT


def make_agent(responses, find, select, book):
    return SchedulingAgent(client=fake_client(responses), model="booking-test-model", prompt_renderer=PromptRenderer(), find_appointment_slots_handler=find, select_appointment_slot_handler=select, book_selected_appointment_handler=book)


@pytest.mark.asyncio
async def test_ambiguous_reply_in_mock_workflow_makes_no_booking_tool_call() -> None:
    client = FakeBookingScheduler()
    _, _, find, select, book, factory = setup_booking(client)
    responses = FakeResponses([output(text="Would you like me to book this exact appointment?")])
    result = await make_agent(responses, find, select, book).respond(context=PromptContext(selected_slot=slot()), user_message="Maybe, I guess.")
    assert result == "Would you like me to book this exact appointment?"
    assert len(responses.calls) == 1
    assert client.book_calls == client.find_calls == []
    factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed", [True, False])
async def test_booking_model_call_receives_function_call_output_before_final_text(confirmed) -> None:
    client = FakeBookingScheduler()
    state, store, find, select, book, _ = setup_booking(client)
    final = "Your appointment is confirmed." if confirmed else "I have not booked that appointment."
    responses = FakeResponses([output("book_selected_appointment", {"confirmed_by_caller": confirmed}), output(text=final)])
    context = PromptContext(customer_verified=True, selected_pet="Milo", selected_slot=slot(), recent_messages=[{"role": "assistant", "content": "Milo at clinic-1 on October 20, 9 a.m. UTC−04:00. Would you like me to book this appointment?"}])
    assert await make_agent(responses, find, select, book).respond(context=context, user_message="Yes." if confirmed else "No.") == final
    assert len(responses.calls) == 2
    for request in responses.calls:
        assert request["tools"] == [FIND_APPOINTMENT_SLOTS_TOOL, SELECT_APPOINTMENT_SLOT_TOOL, BOOK_SELECTED_APPOINTMENT_TOOL]
    result = responses.calls[1]["input"][-1]
    assert result["type"] == "function_call_output"
    assert result["call_id"] == "book_selected_appointment-call"
    assert json.loads(result["output"])["status"] == ("booking_confirmed" if confirmed else "booking_declined")
    assert "private-key-1" not in str(responses.calls)
    assert store.get(state.conversation_id).phase == (ConversationPhase.BOOKING_COMPLETE if confirmed else ConversationPhase.SELECTING_APPOINTMENT)


def test_booking_prompt_requires_exact_confirmation_and_explains_uncertain_outcomes() -> None:
    prompt = PromptRenderer().render_scheduling(PromptContext(selected_pet="Milo", selected_slot=slot()), "Maybe.")
    assert "clinic identifier" in prompt
    assert "clinic clinic-1" in prompt
    assert "only after the caller explicitly says yes" in prompt
    assert "ask a brief clarification instead of calling the booking tool" in prompt
    assert "Describe booking success only when the tool returns booking_confirmed" in prompt
    assert "do not claim failure or success" in prompt
    assert "do not claim success or a definite failure" in prompt
    assert "Never retry automatically" in prompt
    assert "cannot book" not in prompt.lower()


@pytest.mark.asyncio
async def test_lost_response_after_mock_scheduler_booking_retries_same_key_without_double_booking(monkeypatch) -> None:
    secret = "test-scheduler-secret-at-least-32-bytes"
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", secret)
    scheduling_api.reset_booking_store()

    class LoseFirstBookingResponse(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.ASGITransport(app=scheduling_api.app)
            self.keys = []

        async def handle_async_request(self, request):
            response = await self.inner.handle_async_request(request)
            if request.url.path == "/bookings":
                self.keys.append(request.headers["Idempotency-Key"])
                if len(self.keys) == 1:
                    assert response.status_code == 201
                    await response.aclose()
                    raise httpx.ReadTimeout("Simulated lost response after booking", request=request)
            return response

    transport = LoseFirstBookingResponse()
    client = SchedulerClient(base_url="http://scheduler.test", jwt_secret=secret, transport=transport)
    try:
        available = await client.find_slots("pet-1", AppointmentType.ROUTINE)
        state, store, _, _, book, factory = setup_booking(client, chosen=available[0])
        first = await book({"confirmed_by_caller": True})
        assert first == {"status": "scheduler_unavailable", "retry_allowed": True}
        assert len(scheduling_api.BOOKING_STORE._idempotency_store) == 1
        retry_state = store.get(state.conversation_id)
        retry = create_book_selected_appointment_handler(client, conversation_state=retry_state, conversation_store=store, idempotency_key_factory=factory)
        result = await retry({"confirmed_by_caller": True})
        assert result["status"] == "booking_confirmed"
        assert transport.keys == ["private-key-1", "private-key-1"]
        assert len(scheduling_api.BOOKING_STORE._idempotency_store) == 1
        stored_confirmation = scheduling_api.BOOKING_STORE._idempotency_store["private-key-1"][1]
        assert result["confirmation"] == stored_confirmation.model_dump(mode="json")
        assert store.get(state.conversation_id).booking_confirmation == stored_confirmation
        factory.assert_called_once_with()
    finally:
        scheduling_api.reset_booking_store()
