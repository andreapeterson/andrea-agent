"""Appointment and booking domain models for PawLine."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator


class AppointmentType(str, Enum):
    SAME_DAY = "same_day"
    ROUTINE = "routine"


class AppointmentSlot(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    slot_id: str
    clinic_id: str
    starts_at: datetime
    ends_at: datetime
    appointment_type: AppointmentType

    @model_validator(mode="after")
    def validate_slot_timing(self) -> "AppointmentSlot":
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be after starts_at")
        return self


class BookingRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    slot_id: str
    pet_id: str
    confirmed_by_caller: bool


class BookingStatus(str, Enum):
    CONFIRMED = "confirmed"


class BookingConfirmation(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    booking_id: str
    slot_id: str
    pet_id: str
    status: BookingStatus
