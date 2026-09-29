"""Turn-understanding models for one caller message."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TurnIntent(str, Enum):
    PROVIDE_PHONE = "provide_phone"
    SELECT_PET = "select_pet"
    DESCRIBE_CONCERN = "describe_concern"
    ANSWER_INTAKE = "answer_intake"
    ASK_POLICY = "ask_policy"
    SELECT_APPOINTMENT = "select_appointment"
    CONFIRM_BOOKING = "confirm_booking"
    OTHER = "other"


class ExtractedIntakeUpdates(BaseModel):
    """Facts extracted from a single new caller message."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    difficulty_breathing: bool | None = None
    uncontrolled_bleeding: bool | None = None
    collapsed_or_unresponsive: bool | None = None
    known_toxin_exposure: bool | None = None
    rapidly_worsening: bool | None = None


class TurnUnderstanding(BaseModel):
    """Structured interpretation of one human turn."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    intents: list[TurnIntent] = Field(...)
    phone_number: str | None = Field(..., min_length=1)
    pet_reference: str | None = Field(..., min_length=1)
    original_concern: str | None = Field(..., min_length=1)
    intake_updates: ExtractedIntakeUpdates = Field(...)
    policy_question: str | None = Field(..., min_length=1)
    appointment_selection: str | None = Field(..., min_length=1)
    booking_confirmed: bool | None = Field(...)


__all__ = [
    "ExtractedIntakeUpdates",
    "TurnIntent",
    "TurnUnderstanding",
]
