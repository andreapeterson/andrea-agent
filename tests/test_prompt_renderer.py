"""Check that prompt context and templates render as intended."""

from datetime import datetime

import pytest

from app.agents import (
    FRONT_DESK_PROMPT_VERSION,
    PromptContext,
    PromptRenderer,
    PromptRenderingError,
)
from app.models.appointment import AppointmentSlot, AppointmentType
from app.models.conversation import ConversationMessage, ConversationRole
from app.models.routing import IntakeAnswers, RoutingAction, RoutingDecision, RoutingLevel


def make_prompt_context() -> PromptContext:
    """Build representative prompt data for the renderer tests."""
    return PromptContext(
        active_agent="front_desk",
        customer_verified=True,
        customer_first_name="Avery",
        known_pets=["Milo", "Juniper"],
        selected_pet="Milo",
        original_concern="Milo needs a routine checkup.",
        intake_answers=IntakeAnswers(
            difficulty_breathing=False,
            uncontrolled_bleeding=True,
            collapsed_or_unresponsive=None,
            known_toxin_exposure=False,
            rapidly_worsening=None,
        ),
        missing_intake_answers=["collapsed_or_unresponsive", "rapidly_worsening"],
        routing_result=RoutingDecision(
            routing_level=RoutingLevel.NEEDS_MORE_INFORMATION,
            next_action=RoutingAction.ASK_INTAKE_QUESTION,
            missing_fields=["collapsed_or_unresponsive"],
        ),
        offered_slots=[
            AppointmentSlot(
                slot_id="slot-routine-1",
                clinic_id="clinic-1",
                starts_at=datetime(2026, 10, 5, 10, 0),
                ends_at=datetime(2026, 10, 5, 10, 45),
                appointment_type=AppointmentType.ROUTINE,
            )
        ],
        selected_slot=None,
        booking_confirmed=False,
        recent_messages=[
            ConversationMessage(role=ConversationRole.USER, content="Can you find a routine visit?")
        ],
        available_tools=["search_slots"],
    )


def test_prompt_context_renders_and_includes_versioned_sections() -> None:
    prompt = PromptRenderer().render_front_desk(make_prompt_context())

    assert f"`{FRONT_DESK_PROMPT_VERSION}`" in prompt
    assert "## Identity" in prompt
    assert "## Goal" in prompt
    assert "## Personality" in prompt
    assert "## Known conversation context" in prompt
    assert "## Conversation flow" in prompt
    assert "## Tool guidance" in prompt
    assert "## Escalation guidance" in prompt
    assert "## Response requirements" in prompt
    assert "Customer first name: Avery" in prompt
    assert "Known pets: Milo, Juniper" in prompt
    assert "Original concern: Milo needs a routine checkup." in prompt
    assert "slot-routine-1" in prompt
    assert "Available tools: search_slots." in prompt


def test_true_false_and_unknown_context_values_render_distinctly() -> None:
    context = make_prompt_context()
    context.customer_verified = None
    prompt = PromptRenderer().render_front_desk(context)

    assert "Customer verified: unknown" in prompt
    assert "difficulty_breathing: false" in prompt
    assert "uncontrolled_bleeding: true" in prompt
    assert "collapsed_or_unresponsive: unknown" in prompt
    assert "Booking confirmed: false" in prompt


def test_missing_template_variable_fails_loudly(tmp_path) -> None:
    (tmp_path / "front_desk.md.j2").write_text("Missing value: {{ required_value }}\n", encoding="utf-8")

    with pytest.raises(PromptRenderingError, match="Unable to render prompt version"):
        PromptRenderer(template_directory=tmp_path).render_front_desk(make_prompt_context())


def test_caller_jinja_syntax_is_rendered_as_plain_conversation_text() -> None:
    context = make_prompt_context()
    caller_text = "Please repeat this literally: {{ something }}"
    context.recent_messages = [ConversationMessage(role=ConversationRole.USER, content=caller_text)]

    prompt = PromptRenderer().render_front_desk(context)

    assert f"user: {caller_text}" in prompt
    assert "something" in prompt
    assert "UndefinedError" not in prompt


def test_default_context_uses_unknown_and_empty_collections() -> None:
    context = PromptContext()

    assert context.customer_verified is None
    assert context.customer_first_name is None
    assert context.known_pets == []
    assert context.intake_answers.difficulty_breathing is None
    assert context.offered_slots == []
    assert context.booking_confirmed is None
    assert context.available_tools == []