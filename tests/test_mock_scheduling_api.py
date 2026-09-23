from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from mock_services.scheduling_api import app, reset_booking_store

TEST_SECRET = "test-scheduler-secret"


def _token(*, expired: bool = False, bad_signature: bool = False, wrong_issuer: bool = False, wrong_audience: bool = False, wrong_subject: bool = False) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "iss": "pawline",
        "aud": "mock-scheduler",
        "sub": "pawline-service",
        "exp": now + timedelta(minutes=5),
    }
    if expired:
        payload["exp"] = now - timedelta(minutes=5)
    if wrong_issuer:
        payload["iss"] = "someone-else"
    if wrong_audience:
        payload["aud"] = "wrong-audience"
    if wrong_subject:
        payload["sub"] = "not-the-service"

    signing_secret = TEST_SECRET if not bad_signature else "wrong-secret"
    return jwt.encode(payload, signing_secret, algorithm="HS256")


def _auth_headers(token: str | None = None, idempotency_key: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


@pytest.fixture(autouse=True)
def reset_state() -> None:
    reset_booking_store()


def test_valid_jwt_can_retrieve_json_slots(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)
    token = _token()

    response = client.get(
        "/slots?pet_id=pet_2001&appointment_type=same_day",
        headers=_auth_headers(token),
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["appointment_type"] == "same_day"
    assert payload[0]["slot_id"] == "slot_same_day_01"


def test_filtering_by_appointment_type(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)
    token = _token()

    response = client.get(
        "/slots?pet_id=pet_2001&appointment_type=routine",
        headers=_auth_headers(token),
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert all(item["appointment_type"] == "routine" for item in payload)


def test_missing_token_returns_401(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)

    response = client.get("/slots?pet_id=pet_2001&appointment_type=same_day")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing token"


def test_invalid_token_returns_401(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)

    response = client.get(
        "/slots?pet_id=pet_2001&appointment_type=same_day",
        headers={"Authorization": "Bearer bad-token"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing token"


def test_expired_token_returns_401(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)
    token = _token(expired=True)

    response = client.get(
        "/slots?pet_id=pet_2001&appointment_type=same_day",
        headers=_auth_headers(token),
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing token"


def test_missing_query_parameter_returns_422(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)
    token = _token()

    response = client.get(
        "/slots?pet_id=pet_2001",
        headers=_auth_headers(token),
    )

    assert response.status_code == 422


def test_confirmed_booking_returns_201_and_json(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)
    token = _token()

    response = client.post(
        "/bookings",
        json={
            "slot_id": "slot_same_day_01",
            "pet_id": "pet_2001",
            "confirmed_by_caller": True,
        },
        headers=_auth_headers(token, "booking-1"),
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["booking_id"] == "booking_slot_same_day_01_pet_2001"
    assert payload["slot_id"] == "slot_same_day_01"
    assert payload["pet_id"] == "pet_2001"
    assert payload["status"] == "confirmed"


def test_nonexistent_slot_returns_404(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)
    token = _token()

    response = client.post(
        "/bookings",
        json={
            "slot_id": "does_not_exist",
            "pet_id": "pet_2001",
            "confirmed_by_caller": True,
        },
        headers=_auth_headers(token, "booking-2"),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Slot not found"


def test_booking_without_caller_confirmation_returns_400(monkeypatch) -> None:
    monkeypatch.setenv("MOCK_SCHEDULER_JWT_SECRET", TEST_SECRET)
    client = TestClient(app)
    token = _token()

    response = client.post(
        "/bookings",
        json={
            "slot_id": "slot_same_day_01",
            "pet_id": "pet_2001",
            "confirmed_by_caller": False,
        },
        headers=_auth_headers(token, "booking-3"),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "caller-confirmation-required"
