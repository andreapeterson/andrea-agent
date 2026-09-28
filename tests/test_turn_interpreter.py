import asyncio
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.integrations.turn_interpreter import (
    OpenAITurnInterpreter,
    TurnInterpretationRequestError,
    TurnInterpretationResponseError,
    build_turn_interpretation_prompt,
)
from app.models.agent_turn import ExtractedIntakeUpdates, TurnIntent, TurnUnderstanding
from app.models.appointment import AppointmentSlot, AppointmentType
from app.models.conversation import ConversationMessage, ConversationPhase, ConversationRole, ConversationState
from app.models.customer import Customer, Pet, PetSpecies
from app.models.routing import IntakeAnswers


def test_turn_understanding_fields_are_required_and_null_optional_values_are_allowed() -> None:
    payload = {
        "intents": [TurnIntent.PROVIDE_PHONE],
        "phone_number": "321-555-0100",
        "pet_reference": None,
        "original_concern": None,
        "intake_updates": {
            "difficulty_breathing": None,
            "uncontrolled_bleeding": None,
            "collapsed_or_unresponsive": None,
            "known_toxin_exposure": None,
            "rapidly_worsening": None,
        },
        "policy_question": None,
        "appointment_selection": None,
        "booking_confirmed": None,
    }
    understanding = TurnUnderstanding.model_validate(payload)
    assert understanding.intents == [TurnIntent.PROVIDE_PHONE]
    assert understanding.phone_number == "321-555-0100"
    assert understanding.pet_reference is None
    assert understanding.booking_confirmed is None

    with pytest.raises(ValidationError):
        TurnUnderstanding.model_validate({
            "intents": [],
            "pet_reference": None,
            "original_concern": None,
            "intake_updates": {
                "difficulty_breathing": None,
                "uncontrolled_bleeding": None,
                "collapsed_or_unresponsive": None,
                "known_toxin_exposure": None,
                "rapidly_worsening": None,
            },
            "policy_question": None,
            "appointment_selection": None,
            "booking_confirmed": None,
        })


def test_unknown_fields_are_rejected_and_intake_updates_are_separate_objects() -> None:
    with pytest.raises(ValidationError):
        TurnUnderstanding.model_validate({
            "intents": [TurnIntent.OTHER],
            "phone_number": None,
            "pet_reference": None,
            "original_concern": None,
            "intake_updates": {
                "difficulty_breathing": True,
            },
            "policy_question": None,
            "appointment_selection": None,
            "booking_confirmed": None,
            "routing_level": "urgent",
        })

    updates_one = ExtractedIntakeUpdates(difficulty_breathing=True)
    updates_two = ExtractedIntakeUpdates(difficulty_breathing=False)
    assert updates_one is not updates_two
    assert updates_one.difficulty_breathing is True
    assert updates_two.difficulty_breathing is False


def test_message_may_contain_multiple_intents_and_invalid_intent_strings_are_rejected() -> None:
    payload = {
        "intents": [TurnIntent.SELECT_PET, TurnIntent.DESCRIBE_CONCERN, TurnIntent.ANSWER_INTAKE],
        "phone_number": None,
        "pet_reference": "Baxter",
        "original_concern": "He is having trouble breathing.",
        "intake_updates": {
            "difficulty_breathing": True,
            "uncontrolled_bleeding": None,
            "collapsed_or_unresponsive": None,
            "known_toxin_exposure": None,
            "rapidly_worsening": None,
        },
        "policy_question": None,
        "appointment_selection": None,
        "booking_confirmed": None,
    }
    understanding = TurnUnderstanding.model_validate(payload)
    assert understanding.intents == [
        TurnIntent.SELECT_PET,
        TurnIntent.DESCRIBE_CONCERN,
        TurnIntent.ANSWER_INTAKE,
    ]

    with pytest.raises(ValidationError):
        TurnUnderstanding.model_validate({
            "intents": ["not_an_intent"],
            "phone_number": None,
            "pet_reference": None,
            "original_concern": None,
            "intake_updates": {
                "difficulty_breathing": None,
                "uncontrolled_bleeding": None,
                "collapsed_or_unresponsive": None,
                "known_toxin_exposure": None,
                "rapidly_worsening": None,
            },
            "policy_question": None,
            "appointment_selection": None,
            "booking_confirmed": None,
        })


def test_prompt_includes_context_and_excludes_secrets_and_unrelated_data() -> None:
    customer = Customer(
        customer_id="customer_1001",
        first_name="Andrea",
        last_name="Peterson",
        phone_number="3212222222",
        pets=[Pet(pet_id="pet_2001", name="Baxter", species=PetSpecies.DOG)],
    )
    state = ConversationState(
        conversation_id="conv-123",
        phase=ConversationPhase.COLLECTING_INTAKE,
        verified_customer=customer,
        selected_pet_id="pet_2001",
        original_concern="Trouble breathing",
        intake_answers=IntakeAnswers(difficulty_breathing=True),
        offered_slots=[
            AppointmentSlot(
                slot_id="slot_5",
                clinic_id="clinic_1",
                starts_at=datetime(2026, 9, 27, 10, 0, 0),
                ends_at=datetime(2026, 9, 27, 10, 30, 0),
                appointment_type=AppointmentType.SAME_DAY,
            )
        ],
        messages=[
            ConversationMessage(role=ConversationRole.USER, content="hi"),
            ConversationMessage(role=ConversationRole.ASSISTANT, content="What is going on?"),
            ConversationMessage(role=ConversationRole.USER, content="hi again"),
            ConversationMessage(role=ConversationRole.ASSISTANT, content="Tell me what happened."),
            ConversationMessage(role=ConversationRole.USER, content="I have more detail"),
            ConversationMessage(role=ConversationRole.ASSISTANT, content="Keep going."),
            ConversationMessage(role=ConversationRole.USER, content="This is the last part"),
        ],
    )

    prompt = build_turn_interpretation_prompt(state, "My dog is having trouble breathing.")

    assert "Current conversation phase:" in prompt
    assert "COLLECTING_INTAKE" in prompt or "collecting_intake" in prompt
    assert "pet_2001" in prompt
    assert "Baxter" in prompt
    assert "slot_5" in prompt
    assert "difficulty_breathing=True" in prompt
    assert "My dog is having trouble breathing." in prompt
    assert "This is the last part" in prompt
    assert "Do not choose an application action or tool" in prompt
    assert "Do not decide whether a case is urgent" in prompt
    assert "untrusted" in prompt.lower()
    assert "JWT" not in prompt
    assert "API key" not in prompt.lower()
    assert "HMAC" not in prompt
    assert "routing_level" not in prompt
    assert "tool_name" not in prompt


def test_openai_turn_interpreter_uses_structured_outputs_and_keeps_state_unchanged() -> None:
    class FakeResponse:
        def __init__(self, payload: TurnUnderstanding | dict) -> None:
            self.output_parsed = payload

    class FakeResponses:
        def __init__(self) -> None:
            self.model = None
            self.input_text = None
            self.instructions = None
            self.text_format = None

        def parse(self, **kwargs):
            self.model = kwargs["model"]
            self.input_text = kwargs["input"]
            self.instructions = kwargs["instructions"]
            self.text_format = kwargs["text_format"]
            return FakeResponse(
                TurnUnderstanding(
                    intents=[TurnIntent.DESCRIBE_CONCERN, TurnIntent.ANSWER_INTAKE],
                    phone_number=None,
                    pet_reference="Baxter",
                    original_concern="He is having trouble breathing.",
                    intake_updates=ExtractedIntakeUpdates(difficulty_breathing=True),
                    policy_question=None,
                    appointment_selection=None,
                    booking_confirmed=None,
                )
            )

    class FakeClient:
        def __init__(self) -> None:
            self.responses = FakeResponses()

    state = ConversationState(
        conversation_id="conv-abc",
        phase=ConversationPhase.COLLECTING_INTAKE,
        messages=[ConversationMessage(role=ConversationRole.USER, content="Hello")],
    )
    snapshot = state.model_copy(deep=True)

    interpreter = OpenAITurnInterpreter(client=FakeClient(), model="gpt-6-luna")
    result = asyncio.run(interpreter.interpret(state, "It's Baxter. He can't breathe normally."))

    assert result.intents == [TurnIntent.DESCRIBE_CONCERN, TurnIntent.ANSWER_INTAKE]
    assert result.pet_reference == "Baxter"
    assert result.intake_updates.difficulty_breathing is True
    assert state == snapshot
    assert state is not snapshot

    assert isinstance(interpreter._client.responses.model, str)
    assert "Baxter" in interpreter._client.responses.input_text
    assert interpreter._client.responses.text_format is TurnUnderstanding


def test_openai_turn_interpreter_rejects_blank_input_and_handles_bad_responses() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.responses = self

        def parse(self, **kwargs):
            raise RuntimeError("provider exploded")

    state = ConversationState(conversation_id="conv-xy")
    interpreter = OpenAITurnInterpreter(client=FakeClient())

    with pytest.raises(TurnInterpretationRequestError):
        asyncio.run(interpreter.interpret(state, "   "))

    class FakeClientFailure:
        def __init__(self) -> None:
            self.responses = self

        def parse(self, **kwargs):
            return type("Response", (), {"output_parsed": None})()

    interpreter = OpenAITurnInterpreter(client=FakeClientFailure())
    with pytest.raises(TurnInterpretationResponseError):
        asyncio.run(interpreter.interpret(state, "Can you help?"))

    class FakeClientMalformed:
        def __init__(self) -> None:
            self.responses = self

        def parse(self, **kwargs):
            return type("Response", (), {"output_parsed": {"intents": ["bad"]}})()

    interpreter = OpenAITurnInterpreter(client=FakeClientMalformed())
    with pytest.raises(TurnInterpretationResponseError):
        asyncio.run(interpreter.interpret(state, "Can you help?"))


def test_example_behaviors_for_phone_pet_and_booking_context() -> None:
    payload = {
        "intents": [TurnIntent.PROVIDE_PHONE],
        "phone_number": "321-555-0100",
        "pet_reference": None,
        "original_concern": None,
        "intake_updates": {
            "difficulty_breathing": None,
            "uncontrolled_bleeding": None,
            "collapsed_or_unresponsive": None,
            "known_toxin_exposure": None,
            "rapidly_worsening": None,
        },
        "policy_question": None,
        "appointment_selection": None,
        "booking_confirmed": None,
    }
    interpretation = TurnUnderstanding.model_validate(payload)
    assert interpretation.intents == [TurnIntent.PROVIDE_PHONE]
    assert interpretation.phone_number == "321-555-0100"

    state = ConversationState(
        conversation_id="conv-42",
        phase=ConversationPhase.CONFIRMING_BOOKING,
        selected_pet_id="pet_2001",
        offered_slots=[
            AppointmentSlot(
                slot_id="slot_9",
                clinic_id="clinic_1",
                starts_at=datetime(2026, 9, 27, 11, 0, 0),
                ends_at=datetime(2026, 9, 27, 11, 30, 0),
                appointment_type=AppointmentType.SAME_DAY,
            )
        ],
    )
    booking_understanding = TurnUnderstanding.model_validate({
        "intents": [TurnIntent.CONFIRM_BOOKING],
        "phone_number": None,
        "pet_reference": None,
        "original_concern": None,
        "intake_updates": {
            "difficulty_breathing": None,
            "uncontrolled_bleeding": None,
            "collapsed_or_unresponsive": None,
            "known_toxin_exposure": None,
            "rapidly_worsening": None,
        },
        "policy_question": None,
        "appointment_selection": None,
        "booking_confirmed": True,
    })
    assert booking_understanding.booking_confirmed is True

    policy_understanding = TurnUnderstanding.model_validate({
        "intents": [TurnIntent.ASK_POLICY],
        "phone_number": None,
        "pet_reference": None,
        "original_concern": None,
        "intake_updates": {
            "difficulty_breathing": None,
            "uncontrolled_bleeding": None,
            "collapsed_or_unresponsive": None,
            "known_toxin_exposure": None,
            "rapidly_worsening": None,
        },
        "policy_question": "What is your cancellation policy?",
        "appointment_selection": None,
        "booking_confirmed": None,
    })
    assert policy_understanding.policy_question == "What is your cancellation policy?"

    ambiguous_understanding = TurnUnderstanding.model_validate({
        "intents": [TurnIntent.ANSWER_INTAKE],
        "phone_number": None,
        "pet_reference": None,
        "original_concern": None,
        "intake_updates": {
            "difficulty_breathing": None,
            "uncontrolled_bleeding": False,
            "collapsed_or_unresponsive": None,
            "known_toxin_exposure": None,
            "rapidly_worsening": None,
        },
        "policy_question": None,
        "appointment_selection": None,
        "booking_confirmed": None,
    })
    assert ambiguous_understanding.intake_updates.uncontrolled_bleeding is False
    assert ambiguous_understanding.intake_updates.known_toxin_exposure is None

    assert TurnIntent.SELECT_PET in TurnIntent
    assert TurnIntent.CONFIRM_BOOKING in TurnIntent
