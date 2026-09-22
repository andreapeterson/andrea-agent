from datetime import datetime, timedelta, timezone

import httpx
import jwt
import pytest

from app.integrations import (
    CallerConfirmationRequiredError,
    SchedulerClient,
    SchedulerRequestError,
    SchedulerResponseError,
    SlotNotFoundError,
)
from app.models.appointment import AppointmentSlot, AppointmentType, BookingConfirmation, BookingRequest, BookingStatus

TEST_SECRET = "test-scheduler-secret"


def _decode_token(token: str) -> dict:
    return jwt.decode(token, TEST_SECRET, algorithms=["HS256"], issuer="pawline", audience="mock-scheduler")


@pytest.mark.asyncio
async def test_find_slots_sends_correct_path_query_and_jwt() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/slots"
        assert request.url.params["pet_id"] == "pet_2001"
        assert request.url.params["appointment_type"] == "same_day"

        auth = request.headers["Authorization"]
        assert auth.startswith("Bearer ")
        token = auth.split(" ", 1)[1]
        payload = _decode_token(token)
        assert payload["sub"] == "pawline-service"
        assert payload["iss"] == "pawline"
        assert payload["aud"] == "mock-scheduler"
        assert "iat" in payload
        assert "exp" in payload
        assert payload["exp"] > payload["iat"]

        return httpx.Response(
            200,
            json=[
                {
                    "slot_id": "slot_same_day_01",
                    "clinic_id": "clinic_01",
                    "starts_at": "2026-09-21T09:00:00+00:00",
                    "ends_at": "2026-09-21T09:30:00+00:00",
                    "appointment_type": "same_day",
                },
                {
                    "slot_id": "slot_same_day_02",
                    "clinic_id": "clinic_02",
                    "starts_at": "2026-09-21T13:00:00+00:00",
                    "ends_at": "2026-09-21T13:30:00+00:00",
                    "appointment_type": "same_day",
                },
            ],
            request=request,
        )

    client = SchedulerClient(
        base_url="https://scheduler.example.com/",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    slots = await client.find_slots("pet_2001", AppointmentType.SAME_DAY)

    assert len(slots) == 2
    assert slots[0].slot_id == "slot_same_day_01"
    assert isinstance(slots[0], AppointmentSlot)


@pytest.mark.asyncio
async def test_find_slots_200_returns_list_of_slots() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "slot_id": "slot_routine_01",
                    "clinic_id": "clinic_01",
                    "starts_at": "2026-09-23T10:00:00+00:00",
                    "ends_at": "2026-09-23T10:45:00+00:00",
                    "appointment_type": "routine",
                }
            ],
            request=request,
        )

    client = SchedulerClient(
        base_url="https://scheduler.example.com",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    slots = await client.find_slots("pet_2001", AppointmentType.ROUTINE)

    assert slots == [
        AppointmentSlot(
            slot_id="slot_routine_01",
            clinic_id="clinic_01",
            starts_at=datetime(2026, 9, 23, 10, 0, tzinfo=timezone.utc),
            ends_at=datetime(2026, 9, 23, 10, 45, tzinfo=timezone.utc),
            appointment_type=AppointmentType.ROUTINE,
        )
    ]


@pytest.mark.asyncio
async def test_book_appointment_sends_correct_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/bookings"
        assert request.headers["Authorization"].startswith("Bearer ")
        assert request.content is not None
        body = request.read()
        assert b'"slot_id":"slot_same_day_01"' in body
        assert b'"pet_id":"pet_2001"' in body
        assert b'"confirmed_by_caller":true' in body
        return httpx.Response(
            201,
            json={
                "booking_id": "booking_1",
                "slot_id": "slot_same_day_01",
                "pet_id": "pet_2001",
                "status": "confirmed",
            },
            request=request,
        )

    client = SchedulerClient(
        base_url="https://scheduler.example.com/",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    booking = await client.book_appointment(
        BookingRequest(
            slot_id="slot_same_day_01",
            pet_id="pet_2001",
            confirmed_by_caller=True,
        )
    )

    assert booking == BookingConfirmation(
        booking_id="booking_1",
        slot_id="slot_same_day_01",
        pet_id="pet_2001",
        status=BookingStatus.CONFIRMED,
    )


@pytest.mark.asyncio
async def test_book_appointment_404_raises_slot_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Slot not found"}, request=request)

    client = SchedulerClient(
        base_url="https://scheduler.example.com",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SlotNotFoundError):
        await client.book_appointment(
            BookingRequest(
                slot_id="missing_slot",
                pet_id="pet_2001",
                confirmed_by_caller=True,
            )
        )


@pytest.mark.asyncio
async def test_book_appointment_caller_confirmation_required_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": "caller-confirmation-required"}, request=request)

    client = SchedulerClient(
        base_url="https://scheduler.example.com",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(CallerConfirmationRequiredError):
        await client.book_appointment(
            BookingRequest(
                slot_id="slot_same_day_01",
                pet_id="pet_2001",
                confirmed_by_caller=False,
            )
        )


@pytest.mark.asyncio
async def test_find_slots_401_raises_scheduler_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Invalid or missing token"}, request=request)

    client = SchedulerClient(
        base_url="https://scheduler.example.com",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SchedulerRequestError):
        await client.find_slots("pet_2001", AppointmentType.SAME_DAY)


@pytest.mark.asyncio
async def test_find_slots_500_raises_scheduler_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Server Error", request=request)

    client = SchedulerClient(
        base_url="https://scheduler.example.com",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SchedulerRequestError):
        await client.find_slots("pet_2001", AppointmentType.SAME_DAY)


@pytest.mark.asyncio
async def test_find_slots_network_failure_raises_scheduler_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection failure", request=request)

    client = SchedulerClient(
        base_url="https://scheduler.example.com",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SchedulerRequestError):
        await client.find_slots("pet_2001", AppointmentType.SAME_DAY)


@pytest.mark.asyncio
async def test_find_slots_malformed_json_raises_scheduler_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json", request=request)

    client = SchedulerClient(
        base_url="https://scheduler.example.com",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SchedulerResponseError):
        await client.find_slots("pet_2001", AppointmentType.SAME_DAY)


@pytest.mark.asyncio
async def test_find_slots_invalid_structure_raises_scheduler_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"clinic_id": "clinic_01"}],
            request=request,
        )

    client = SchedulerClient(
        base_url="https://scheduler.example.com",
        jwt_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(SchedulerResponseError):
        await client.find_slots("pet_2001", AppointmentType.SAME_DAY)
