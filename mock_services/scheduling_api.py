import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models.appointment import (
    AppointmentSlot,
    AppointmentType,
    BookingConfirmation,
    BookingRequest,
    BookingStatus,
)

app = FastAPI(title="Mock Scheduling API")
security = HTTPBearer(auto_error=False)

TEST_SECRET = "dev-scheduler-secret"


def _jwt_secret() -> str:
    return os.getenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)


def _require_valid_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> None:
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            issuer="pawline",
            audience="mock-scheduler",
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")

    if payload.get("sub") != "pawline-service":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")


APPOINTMENT_SLOTS: list[AppointmentSlot] = [
    AppointmentSlot(
        slot_id="slot_same_day_01",
        clinic_id="clinic_01",
        starts_at=datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc),
        appointment_type=AppointmentType.SAME_DAY,
    ),
    AppointmentSlot(
        slot_id="slot_same_day_02",
        clinic_id="clinic_02",
        starts_at=datetime(2026, 9, 21, 13, 0, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 21, 13, 30, tzinfo=timezone.utc),
        appointment_type=AppointmentType.SAME_DAY,
    ),
    AppointmentSlot(
        slot_id="slot_routine_01",
        clinic_id="clinic_01",
        starts_at=datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 23, 10, 45, tzinfo=timezone.utc),
        appointment_type=AppointmentType.ROUTINE,
    ),
    AppointmentSlot(
        slot_id="slot_routine_02",
        clinic_id="clinic_03",
        starts_at=datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc),
        ends_at=datetime(2026, 9, 24, 15, 45, tzinfo=timezone.utc),
        appointment_type=AppointmentType.ROUTINE,
    ),
]


@app.get("/slots", response_model=list[AppointmentSlot])
def get_slots(
    pet_id: str = Query(..., description="Pet identifier."),
    appointment_type: AppointmentType = Query(..., description="Appointment type."),
    _token_ok: None = Depends(_require_valid_token),
) -> list[AppointmentSlot]:
    _ = pet_id #unused for now, but could be used in the future to filter slots based on pet characteristics
    return [slot for slot in APPOINTMENT_SLOTS if slot.appointment_type == appointment_type]


@app.post("/bookings", response_model=BookingConfirmation, status_code=status.HTTP_201_CREATED)
def create_booking(
    request: BookingRequest,
    _token_ok: None = Depends(_require_valid_token),
) -> BookingConfirmation:
    slot = next((slot for slot in APPOINTMENT_SLOTS if slot.slot_id == request.slot_id), None)
    if slot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Slot not found")
    if not request.confirmed_by_caller:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="caller-confirmation-required")

    booking_id = f"booking_{request.slot_id}_{request.pet_id}"
    return BookingConfirmation(
        booking_id=booking_id,
        slot_id=request.slot_id,
        pet_id=request.pet_id,
        status=BookingStatus.CONFIRMED,
    )
