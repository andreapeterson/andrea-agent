import hashlib
import hmac
import json
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from app.integrations.handoff_client import (
    HandoffClient,
    HandoffNotRequiredError,
    HandoffRequestError,
    HandoffResponseError,
)
from app.models.handoff import HandoffReceipt, HandoffRequest, HandoffStatus, HandoffSummary
from app.models.routing import IntakeAnswers, RoutingAction, RoutingDecision, RoutingLevel

TEST_SECRET = "test-handoff-secret"
FIXED_TIMESTAMP = "1700000000"


def _make_handoff_request() -> HandoffRequest:
    return HandoffRequest(
        conversation_id="conv-123",
        verified_customer_id="customer_1001",
        selected_pet_id="pet_2001",
        original_concern="labored breathing",
        intake_answers=IntakeAnswers(
            difficulty_breathing=True,
            uncontrolled_bleeding=False,
            collapsed_or_unresponsive=False,
            known_toxin_exposure=False,
            rapidly_worsening=False,
        ),
        routing_decision=RoutingDecision(
            routing_level=RoutingLevel.URGENT,
            next_action=RoutingAction.CREATE_HANDOFF,
            matched_rule_ids=["urgent-difficulty-breathing"],
            missing_fields=[],
        ),
        summary=HandoffSummary(
            original_concern="labored breathing",
            routing_level=RoutingLevel.URGENT,
            positive_signals=["difficulty_breathing"],
            negative_signals=[
                "uncontrolled_bleeding",
                "collapsed_or_unresponsive",
                "known_toxin_exposure",
                "rapidly_worsening",
            ],
            unanswered_signals=[],
            matched_rule_ids=["urgent-difficulty-breathing"],
        ),
    )


def _expected_signature(secret: str, timestamp: str, body_bytes: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


@pytest.mark.asyncio
async def test_create_handoff_sends_signed_json_and_returns_receipt() -> None:
    handoff_request = _make_handoff_request()
    body_bytes = handoff_request.model_dump_json().encode("utf-8")
    expected_signature = _expected_signature(TEST_SECRET, FIXED_TIMESTAMP, body_bytes)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/handoffs"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["X-PawLine-Timestamp"] == FIXED_TIMESTAMP
        assert request.headers["X-PawLine-Signature"] == expected_signature
        assert request.content == body_bytes
        payload = json.loads(request.content.decode("utf-8"))
        assert HandoffRequest.model_validate(payload)
        return httpx.Response(
            202,
            json={
                "case_id": "case_123",
                "conversation_id": "conv-123",
                "status": "queued",
                "received_at": "2026-09-23T00:00:00+00:00",
            },
            request=request,
        )

    client = HandoffClient(
        base_url="https://handoff.example.com",
        webhook_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with patch("app.integrations.handoff_client.time.time", return_value=float(FIXED_TIMESTAMP)):
        receipt = await client.create_handoff(handoff_request)

    assert receipt == HandoffReceipt(
        case_id="case_123",
        conversation_id="conv-123",
        status=HandoffStatus.QUEUED,
        received_at=datetime(2026, 9, 23, 0, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_create_handoff_409_handoff_not_required_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "handoff-not-required"}, request=request)

    client = HandoffClient(
        base_url="https://handoff.example.com",
        webhook_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with patch("app.integrations.handoff_client.time.time", return_value=float(FIXED_TIMESTAMP)):
        with pytest.raises(HandoffNotRequiredError):
            await client.create_handoff(_make_handoff_request())


@pytest.mark.asyncio
async def test_create_handoff_401_and_500_raise_request_error() -> None:
    for status_code in (401, 500):
        def handler(request: httpx.Request, status_code: int = status_code) -> httpx.Response:
            return httpx.Response(status_code, json={"detail": "error"}, request=request)

        client = HandoffClient(
            base_url="https://handoff.example.com",
            webhook_secret=TEST_SECRET,
            timeout_seconds=2.0,
            transport=httpx.MockTransport(handler),
        )

        with patch("app.integrations.handoff_client.time.time", return_value=float(FIXED_TIMESTAMP)):
            with pytest.raises(HandoffRequestError):
                await client.create_handoff(_make_handoff_request())


@pytest.mark.asyncio
async def test_create_handoff_network_failure_raises_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection failure", request=request)

    client = HandoffClient(
        base_url="https://handoff.example.com",
        webhook_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with patch("app.integrations.handoff_client.time.time", return_value=float(FIXED_TIMESTAMP)):
        with pytest.raises(HandoffRequestError):
            await client.create_handoff(_make_handoff_request())


@pytest.mark.asyncio
async def test_create_handoff_malformed_json_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, text="not-json", request=request)

    client = HandoffClient(
        base_url="https://handoff.example.com",
        webhook_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with patch("app.integrations.handoff_client.time.time", return_value=float(FIXED_TIMESTAMP)):
        with pytest.raises(HandoffResponseError):
            await client.create_handoff(_make_handoff_request())


@pytest.mark.asyncio
async def test_create_handoff_invalid_receipt_schema_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202, json={"status": "queued"}, request=request)

    client = HandoffClient(
        base_url="https://handoff.example.com",
        webhook_secret=TEST_SECRET,
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with patch("app.integrations.handoff_client.time.time", return_value=float(FIXED_TIMESTAMP)):
        with pytest.raises(HandoffResponseError):
            await client.create_handoff(_make_handoff_request())
