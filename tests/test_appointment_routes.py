from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_scheduler_client
from app.integrations import (
    CallerConfirmationRequiredError,
    SchedulerRequestError,
    SchedulerResponseError,
    SlotNotFoundError,
)
from app.main import app
from app.models.appointment import (
    AppointmentSlot,
    AppointmentType,
    BookingConfirmation,
    BookingRequest,
    BookingStatus,
)


class FakeSchedulerClient:
    def __init__(self) -> None:
        self.passed_pet_ids: list[str] = []
        self.passed_appointment_types: list[AppointmentType] = []
        self.booking_requests: list[BookingRequest] = []
        self.passed_idempotency_keys: list[str] = []
        self.slots_to_return: list[AppointmentSlot] = [
            AppointmentSlot(
                slot_id="slot_same_day_01",
                clinic_id="clinic_01",
                starts_at=datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc),
                ends_at=datetime(2026, 9, 21, 9, 30, tzinfo=timezone.utc),
                appointment_type=AppointmentType.SAME_DAY,
            )
        ]
        self.booking_to_return = BookingConfirmation(
            booking_id="booking_101",
            slot_id="slot_same_day_01",
            pet_id="pet_2001",
            status=BookingStatus.CONFIRMED,
        )
        self.find_slots_error: Exception | None = None
        self.book_appointment_error: Exception | None = None

    async def find_slots(self, pet_id: str, appointment_type: AppointmentType) -> list[AppointmentSlot]:
        self.passed_pet_ids.append(pet_id)
        self.passed_appointment_types.append(appointment_type)
        if self.find_slots_error is not None:
            raise self.find_slots_error
        return self.slots_to_return

    async def book_appointment(self, booking_request: BookingRequest, idempotency_key: str) -> BookingConfirmation:
        self.booking_requests.append(booking_request)
        self.passed_idempotency_keys.append(idempotency_key)
        if self.book_appointment_error is not None:
            raise self.book_appointment_error
        return self.booking_to_return


@pytest.fixture
def fake_scheduler_client() -> FakeSchedulerClient:
    return FakeSchedulerClient()


@pytest.fixture
def client(fake_scheduler_client: FakeSchedulerClient) -> TestClient:
    app.dependency_overrides[get_scheduler_client] = lambda: fake_scheduler_client
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_get_appointment_slots_success(client: TestClient, fake_scheduler_client: FakeSchedulerClient) -> None:
    response = client.get("/appointments/slots?pet_id=pet_2001&appointment_type=same_day")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["slot_id"] == "slot_same_day_01"
    assert payload[0]["appointment_type"] == "same_day"
    assert fake_scheduler_client.passed_pet_ids == ["pet_2001"]
    assert fake_scheduler_client.passed_appointment_types == [AppointmentType.SAME_DAY]


def test_get_appointment_slots_missing_pet_id(client: TestClient) -> None:
    response = client.get("/appointments/slots?appointment_type=same_day")

    assert response.status_code == 422


def test_get_appointment_slots_invalid_appointment_type(client: TestClient) -> None:
    response = client.get("/appointments/slots?pet_id=pet_2001&appointment_type=not-valid")

    assert response.status_code == 422


def test_get_appointment_slots_scheduler_response_error(client: TestClient, fake_scheduler_client: FakeSchedulerClient) -> None:
    fake_scheduler_client.find_slots_error = SchedulerResponseError("bad data")

    response = client.get("/appointments/slots?pet_id=pet_2001&appointment_type=same_day")

    assert response.status_code == 502
    assert response.json()["detail"] == "invalid-scheduler-data"


def test_get_appointment_slots_scheduler_request_error(client: TestClient, fake_scheduler_client: FakeSchedulerClient) -> None:
    fake_scheduler_client.find_slots_error = SchedulerRequestError("scheduler down")

    response = client.get("/appointments/slots?pet_id=pet_2001&appointment_type=same_day")

    assert response.status_code == 503
    assert response.json()["detail"] == "scheduling-service-unavailable"


def test_post_appointment_booking_success(client: TestClient, fake_scheduler_client: FakeSchedulerClient) -> None:
    response = client.post(
        "/appointments/bookings",
        json={
            "slot_id": "slot_same_day_01",
            "pet_id": "pet_2001",
            "confirmed_by_caller": True,
        },
        headers={"Idempotency-Key": "req-123"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["booking_id"] == "booking_101"
    assert payload["status"] == "confirmed"
    assert fake_scheduler_client.booking_requests[0] == BookingRequest(
        slot_id="slot_same_day_01",
        pet_id="pet_2001",
        confirmed_by_caller=True,
    )
    assert fake_scheduler_client.passed_idempotency_keys == ["req-123"]


def test_post_appointment_booking_missing_or_invalid_body_fields(client: TestClient) -> None:
    response = client.post("/appointments/bookings", json={})
    assert response.status_code == 422

    response = client.post(
        "/appointments/bookings",
        json={
            "slot_id": "slot_same_day_01",
            "pet_id": "pet_2001",
            "confirmed_by_caller": {"truthy": True},
        },
    )
    assert response.status_code == 422


def test_post_appointment_booking_slot_not_found(client: TestClient, fake_scheduler_client: FakeSchedulerClient) -> None:
    fake_scheduler_client.book_appointment_error = SlotNotFoundError("Slot not found")

    response = client.post(
        "/appointments/bookings",
        json={
            "slot_id": "missing_slot",
            "pet_id": "pet_2001",
            "confirmed_by_caller": True,
        },
        headers={"Idempotency-Key": "req-123"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Slot not found"


def test_post_appointment_booking_confirmation_required(client: TestClient, fake_scheduler_client: FakeSchedulerClient) -> None:
    fake_scheduler_client.book_appointment_error = CallerConfirmationRequiredError("caller confirmation required")

    response = client.post(
        "/appointments/bookings",
        json={
            "slot_id": "slot_same_day_01",
            "pet_id": "pet_2001",
            "confirmed_by_caller": False,
        },
        headers={"Idempotency-Key": "req-123"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "caller-confirmation-required"


def test_post_appointment_booking_scheduler_response_error(client: TestClient, fake_scheduler_client: FakeSchedulerClient) -> None:
    fake_scheduler_client.book_appointment_error = SchedulerResponseError("bad data")

    response = client.post(
        "/appointments/bookings",
        json={
            "slot_id": "slot_same_day_01",
            "pet_id": "pet_2001",
            "confirmed_by_caller": True,
        },
        headers={"Idempotency-Key": "req-123"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "invalid-scheduler-data"


def test_post_appointment_booking_scheduler_request_error(client: TestClient, fake_scheduler_client: FakeSchedulerClient) -> None:
    fake_scheduler_client.book_appointment_error = SchedulerRequestError("scheduler down")

    response = client.post(
        "/appointments/bookings",
        json={
            "slot_id": "slot_same_day_01",
            "pet_id": "pet_2001",
            "confirmed_by_caller": True,
        },
        headers={"Idempotency-Key": "req-123"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "scheduling-service-unavailable"
