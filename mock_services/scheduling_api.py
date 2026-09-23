import os
from datetime import datetime, timezone
from threading import Lock

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
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


class BookingStore:
    def __init__(self) -> None:
        self._all_slots: dict[str, AppointmentSlot] = {
            "slot_same_day_01": AppointmentSlot(
                slot_id="slot_same_day_01",
                clinic_id="clinic_01",
                starts_at=datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc),
                appointment_type=AppointmentType.SAME_DAY,
            ),
            "slot_same_day_02": AppointmentSlot(
                slot_id="slot_same_day_02",
                clinic_id="clinic_02",
                starts_at=datetime(2026, 9, 21, 13, 0, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 21, 13, 30, tzinfo=timezone.utc),
                appointment_type=AppointmentType.SAME_DAY,
            ),
            "slot_routine_01": AppointmentSlot(
                slot_id="slot_routine_01",
                clinic_id="clinic_01",
                starts_at=datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 23, 10, 45, tzinfo=timezone.utc),
                appointment_type=AppointmentType.ROUTINE,
            ),
            "slot_routine_02": AppointmentSlot(
                slot_id="slot_routine_02",
                clinic_id="clinic_03",
                starts_at=datetime(2026, 9, 24, 15, 0, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 24, 15, 45, tzinfo=timezone.utc),
                appointment_type=AppointmentType.ROUTINE,
            ),
        }
        self._available_slot_ids: set[str] = set(self._all_slots)
        self._idempotency_store: dict[str, tuple[tuple[str, str, bool], BookingConfirmation]] = {}
        self._lock = Lock()

    @staticmethod
    def _fingerprint(request: BookingRequest) -> tuple[str, str, bool]:
        return (request.slot_id, request.pet_id, request.confirmed_by_caller)

    def get_available_slots(self, appointment_type: AppointmentType | None = None) -> list[AppointmentSlot]:
        slots = [self._all_slots[slot_id] for slot_id in sorted(self._available_slot_ids)]
        if appointment_type is not None:
            slots = [slot for slot in slots if slot.appointment_type == appointment_type]
        return slots

    def book(self, idempotency_key: str, request: BookingRequest) -> BookingConfirmation:
        fingerprint = self._fingerprint(request)
        with self._lock:
            existing = self._idempotency_store.get(idempotency_key)
            if existing is not None:
                existing_fingerprint, confirmation = existing
                if existing_fingerprint == fingerprint:
                    return confirmation
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="idempotency-key-reused")

            if request.slot_id not in self._all_slots:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Slot not found")
            if request.slot_id not in self._available_slot_ids:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="slot-unavailable")
            if not request.confirmed_by_caller:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="caller-confirmation-required")

            confirmation = BookingConfirmation(
                booking_id=f"booking_{request.slot_id}_{request.pet_id}",
                slot_id=request.slot_id,
                pet_id=request.pet_id,
                status=BookingStatus.CONFIRMED,
            )
            self._idempotency_store[idempotency_key] = (fingerprint, confirmation)
            self._available_slot_ids.remove(request.slot_id)
            return confirmation


BOOKING_STORE = BookingStore()


def reset_booking_store() -> None:
    global BOOKING_STORE
    BOOKING_STORE = BookingStore()


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


@app.get("/slots", response_model=list[AppointmentSlot])
def get_slots(
    pet_id: str = Query(..., description="Pet identifier."),
    appointment_type: AppointmentType = Query(..., description="Appointment type."),
    _token_ok: None = Depends(_require_valid_token),
) -> list[AppointmentSlot]:
    _ = pet_id #unused for now, but could be used in the future to filter slots based on pet characteristics
    return BOOKING_STORE.get_available_slots(appointment_type)


@app.post("/bookings", response_model=BookingConfirmation, status_code=status.HTTP_201_CREATED)
def create_booking(
    request: BookingRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    _token_ok: None = Depends(_require_valid_token),
) -> BookingConfirmation:
    _ = idempotency_key
    return BOOKING_STORE.book(idempotency_key, request)
