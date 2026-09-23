from datetime import datetime

import pytest
from pydantic import ValidationError

from app.models import (
    AppointmentSlot,
    AppointmentType,
    BookingConfirmation,
    BookingRequest,
    BookingStatus,
)


def test_appointment_slot_valid_model_construction() -> None:
    starts_at = datetime(2026, 1, 10, 9, 0, 0)
    ends_at = datetime(2026, 1, 10, 9, 30, 0)

    slot = AppointmentSlot(
        slot_id="slot_1001",
        clinic_id="clinic_5",
        starts_at=starts_at,
        ends_at=ends_at,
        appointment_type=AppointmentType.SAME_DAY,
    )

    assert slot.slot_id == "slot_1001"
    assert slot.clinic_id == "clinic_5"
    assert slot.starts_at == starts_at
    assert slot.ends_at == ends_at
    assert slot.appointment_type == AppointmentType.SAME_DAY


def test_appointment_slot_datetime_fields_convert_to_json() -> None:
    starts_at = datetime(2026, 1, 10, 9, 0, 0)
    ends_at = datetime(2026, 1, 10, 9, 30, 0)

    slot = AppointmentSlot(
        slot_id="slot_1001",
        clinic_id="clinic_5",
        starts_at=starts_at,
        ends_at=ends_at,
        appointment_type=AppointmentType.ROUTINE,
    )

    payload = slot.model_dump(mode="json")

    assert payload["starts_at"] == "2026-01-10T09:00:00"
    assert payload["ends_at"] == "2026-01-10T09:30:00"
    assert payload["appointment_type"] == "routine"


def test_appointment_slot_rejects_unsupported_appointment_type() -> None:
    with pytest.raises(ValidationError):
        AppointmentSlot(
            slot_id="slot_1001",
            clinic_id="clinic_5",
            starts_at=datetime(2026, 1, 10, 9, 0, 0),
            ends_at=datetime(2026, 1, 10, 9, 30, 0),
            appointment_type="urgent",
        )


def test_appointment_slot_rejects_ends_at_not_after_starts_at() -> None:
    with pytest.raises(ValidationError):
        AppointmentSlot(
            slot_id="slot_1001",
            clinic_id="clinic_5",
            starts_at=datetime(2026, 1, 10, 9, 30, 0),
            ends_at=datetime(2026, 1, 10, 9, 30, 0),
            appointment_type=AppointmentType.SAME_DAY,
        )

    with pytest.raises(ValidationError):
        AppointmentSlot(
            slot_id="slot_1001",
            clinic_id="clinic_5",
            starts_at=datetime(2026, 1, 10, 9, 45, 0),
            ends_at=datetime(2026, 1, 10, 9, 30, 0),
            appointment_type=AppointmentType.SAME_DAY,
        )


def test_booking_request_and_confirmation_valid_models() -> None:
    request = BookingRequest(
        slot_id="slot_1001",
        pet_id="pet_2001",
        confirmed_by_caller=True,
    )

    confirmation = BookingConfirmation(
        booking_id="booking_9001",
        slot_id="slot_1001",
        pet_id="pet_2001",
        status=BookingStatus.CONFIRMED,
    )

    assert request.slot_id == "slot_1001"
    assert request.pet_id == "pet_2001"
    assert request.confirmed_by_caller is True

    assert confirmation.booking_id == "booking_9001"
    assert confirmation.slot_id == "slot_1001"
    assert confirmation.pet_id == "pet_2001"
    assert confirmation.status == BookingStatus.CONFIRMED
