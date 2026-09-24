import hashlib
import hmac
import json
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from mock_services.handoff_api import app

TEST_SECRET = "test-handoff-secret"


def _signed_request_bytes(payload: dict, *, secret: str = TEST_SECRET, timestamp: int | None = None) -> tuple[bytes, dict[str, str]]:
    raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ts = str(int(time.time()) if timestamp is None else timestamp)
    signature = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        ts.encode("utf-8") + b"." + raw_body,
        hashlib.sha256,
    ).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-PawLine-Timestamp": ts,
        "X-PawLine-Signature": signature,
    }
    return raw_body, headers


def _urgent_payload() -> dict:
    return {
        "conversation_id": "conv-123",
        "verified_customer_id": "cust-123",
        "selected_pet_id": "pet-456",
        "original_concern": "labored breathing",
        "intake_answers": {
            "difficulty_breathing": True,
            "uncontrolled_bleeding": False,
            "collapsed_or_unresponsive": False,
            "known_toxin_exposure": False,
            "rapidly_worsening": False,
        },
        "routing_decision": {
            "routing_level": "urgent",
            "next_action": "create_handoff",
            "matched_rule_ids": ["urgent-difficulty-breathing"],
            "missing_fields": [],
        },
    }


def _non_urgent_payload() -> dict:
    return {
        "conversation_id": "conv-124",
        "verified_customer_id": "cust-123",
        "selected_pet_id": "pet-456",
        "original_concern": "mild limp",
        "intake_answers": {
            "difficulty_breathing": False,
            "uncontrolled_bleeding": False,
            "collapsed_or_unresponsive": False,
            "known_toxin_exposure": False,
            "rapidly_worsening": False,
        },
        "routing_decision": {
            "routing_level": "routine",
            "next_action": "search_routine_appointment",
            "matched_rule_ids": ["routine-no-escalation-indicators"],
            "missing_fields": [],
        },
    }


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("MOCK_HANDOFF_WEBHOOK_SECRET", TEST_SECRET)
    return TestClient(app)


def test_valid_signed_urgent_handoff_returns_202(client: TestClient) -> None:
    payload = _urgent_payload()
    raw_body, headers = _signed_request_bytes(payload)

    response = client.post("/handoffs", content=raw_body, headers=headers)

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert data["conversation_id"] == "conv-123"
    assert data["case_id"].startswith("case_")
    assert datetime.fromisoformat(data["received_at"]).tzinfo is not None


def test_missing_signature_returns_401(client: TestClient) -> None:
    payload = _urgent_payload()
    raw_body, _ = _signed_request_bytes(payload)

    response = client.post(
        "/handoffs",
        content=raw_body,
        headers={"Content-Type": "application/json", "X-PawLine-Timestamp": str(int(time.time()))},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing signature"


def test_incorrect_signature_returns_401(client: TestClient) -> None:
    payload = _urgent_payload()
    raw_body, headers = _signed_request_bytes(payload)
    headers["X-PawLine-Signature"] = "sha256=deadbeef"

    response = client.post("/handoffs", content=raw_body, headers=headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing signature"


def test_signature_with_wrong_secret_returns_401(client: TestClient) -> None:
    payload = _urgent_payload()
    raw_body, headers = _signed_request_bytes(payload, secret="wrong-secret")

    response = client.post("/handoffs", content=raw_body, headers=headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing signature"


def test_changing_body_after_signing_returns_401(client: TestClient) -> None:
    payload = _urgent_payload()
    raw_body, headers = _signed_request_bytes(payload)
    tampered = b'{"conversation_id":"conv-999"}'

    response = client.post("/handoffs", content=tampered, headers=headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing signature"


def test_malformed_timestamp_returns_401(client: TestClient) -> None:
    payload = _urgent_payload()
    raw_body, _ = _signed_request_bytes(payload)

    response = client.post(
        "/handoffs",
        content=raw_body,
        headers={
            "Content-Type": "application/json",
            "X-PawLine-Timestamp": "not-a-timestamp",
            "X-PawLine-Signature": "sha256=deadbeef",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing signature"


def test_stale_timestamp_returns_401(client: TestClient) -> None:
    payload = _urgent_payload()
    stale_ts = int(time.time()) - 301
    raw_body, headers = _signed_request_bytes(payload, timestamp=stale_ts)

    response = client.post("/handoffs", content=raw_body, headers=headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing signature"


def test_timestamp_too_far_in_future_returns_401(client: TestClient) -> None:
    payload = _urgent_payload()
    future_ts = int(time.time()) + 61
    raw_body, headers = _signed_request_bytes(payload, timestamp=future_ts)

    response = client.post("/handoffs", content=raw_body, headers=headers)

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing signature"


def test_valid_signature_with_invalid_json_structure_returns_422(client: TestClient) -> None:
    payload = {"conversation_id": 999}
    raw_body, headers = _signed_request_bytes(payload)

    response = client.post("/handoffs", content=raw_body, headers=headers)

    assert response.status_code == 422


def test_valid_signed_non_urgent_decision_returns_409(client: TestClient) -> None:
    payload = _non_urgent_payload()
    raw_body, headers = _signed_request_bytes(payload)

    response = client.post("/handoffs", content=raw_body, headers=headers)

    assert response.status_code == 409
    assert response.json()["detail"] == "handoff-not-required"
