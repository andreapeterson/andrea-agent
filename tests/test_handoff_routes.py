from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_handoff_client
from app.integrations import HandoffNotRequiredError, HandoffRequestError, HandoffResponseError
from app.main import app
from app.models.handoff import HandoffCreateRequest, HandoffReceipt, HandoffRequest, HandoffStatus
from app.models.routing import IntakeAnswers, RoutingAction, RoutingLevel


class FakeHandoffClient:
    def __init__(self) -> None:
        self.calls: list[HandoffRequest] = []
        self.error: Exception | None = None

    async def create_handoff(self, handoff_request: HandoffRequest) -> HandoffReceipt:
        self.calls.append(handoff_request)
        if self.error is not None:
            raise self.error
        return HandoffReceipt(
            case_id="case_abc",
            conversation_id=handoff_request.conversation_id,
            status=HandoffStatus.QUEUED,
            received_at=datetime.now(timezone.utc),
        )


@pytest.fixture
def fake_handoff_client() -> FakeHandoffClient:
    return FakeHandoffClient()


@pytest.fixture
def client(fake_handoff_client: FakeHandoffClient) -> TestClient:
    app.dependency_overrides[get_handoff_client] = lambda: fake_handoff_client
    yield TestClient(app)
    app.dependency_overrides.clear()


def _urgent_payload() -> dict:
    return {
        "conversation_id": "conv-123",
        "verified_customer_id": "customer_1001",
        "selected_pet_id": "pet_2001",
        "original_concern": "labored breathing",
        "intake_answers": {
            "difficulty_breathing": True,
            "uncontrolled_bleeding": False,
            "collapsed_or_unresponsive": False,
            "known_toxin_exposure": False,
            "rapidly_worsening": False,
        },
    }


def _routine_payload() -> dict:
    return {
        "conversation_id": "conv-456",
        "verified_customer_id": "customer_1001",
        "selected_pet_id": "pet_2001",
        "original_concern": "mild fatigue",
        "intake_answers": {
            "difficulty_breathing": False,
            "uncontrolled_bleeding": False,
            "collapsed_or_unresponsive": False,
            "known_toxin_exposure": False,
            "rapidly_worsening": False,
        },
    }


def _incomplete_payload() -> dict:
    return {
        "conversation_id": "conv-789",
        "verified_customer_id": "customer_1001",
        "selected_pet_id": "pet_2001",
        "original_concern": "uncertain",
        "intake_answers": {
            "difficulty_breathing": None,
            "uncontrolled_bleeding": False,
            "collapsed_or_unresponsive": False,
            "known_toxin_exposure": False,
            "rapidly_worsening": False,
        },
    }


def test_urgent_intake_returns_202(client: TestClient, fake_handoff_client: FakeHandoffClient) -> None:
    response = client.post("/handoffs", json=_urgent_payload())

    assert response.status_code == 202
    assert fake_handoff_client.calls[0].routing_decision.routing_level == RoutingLevel.URGENT
    assert fake_handoff_client.calls[0].routing_decision.next_action == RoutingAction.CREATE_HANDOFF
    assert fake_handoff_client.calls[0].conversation_id == "conv-123"
    assert response.json()["status"] == "queued"


def test_route_recomputes_routing_decision_and_ignores_caller_routing_decision(client: TestClient) -> None:
    payload = {
        **_urgent_payload(),
        "routing_decision": {
            "routing_level": "routine",
            "next_action": "search_routine_appointment",
            "matched_rule_ids": ["routine-no-escalation-indicators"],
            "missing_fields": [],
        },
    }

    response = client.post("/handoffs", json=payload)

    assert response.status_code == 422


def test_attempting_to_include_routing_decision_in_json_body_returns_422(client: TestClient) -> None:
    payload = {
        **_urgent_payload(),
        "routing_decision": {
            "routing_level": "urgent",
            "next_action": "create_handoff",
            "matched_rule_ids": ["urgent-difficulty-breathing"],
            "missing_fields": [],
        },
    }

    response = client.post("/handoffs", json=payload)

    assert response.status_code == 422


def test_routine_intake_returns_409_handoff_not_required(client: TestClient) -> None:
    response = client.post("/handoffs", json=_routine_payload())

    assert response.status_code == 409
    assert response.json()["detail"] == "handoff-not-required"


def test_incomplete_non_urgent_intake_returns_409(client: TestClient) -> None:
    response = client.post("/handoffs", json=_incomplete_payload())

    assert response.status_code == 409
    assert response.json()["detail"] == "handoff-not-required"


def test_fake_client_is_not_called_when_routing_rejects_handoff(
    client: TestClient,
    fake_handoff_client: FakeHandoffClient,
) -> None:
    response = client.post("/handoffs", json=_routine_payload())

    assert response.status_code == 409
    assert fake_handoff_client.calls == []


def test_handoff_not_required_error_becomes_409(client: TestClient, fake_handoff_client: FakeHandoffClient) -> None:
    fake_handoff_client.error = HandoffNotRequiredError("not required")

    response = client.post("/handoffs", json=_urgent_payload())

    assert response.status_code == 409
    assert response.json()["detail"] == "handoff-not-required"


def test_handoff_response_error_becomes_502(client: TestClient, fake_handoff_client: FakeHandoffClient) -> None:
    fake_handoff_client.error = HandoffResponseError("bad data")

    response = client.post("/handoffs", json=_urgent_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == "invalid-handoff-data"


def test_handoff_request_error_becomes_503(client: TestClient, fake_handoff_client: FakeHandoffClient) -> None:
    fake_handoff_client.error = HandoffRequestError("service unavailable")

    response = client.post("/handoffs", json=_urgent_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "handoff-service-unavailable"


def test_invalid_or_missing_body_fields_return_422(client: TestClient) -> None:
    response = client.post("/handoffs", json={})
    assert response.status_code == 422

    response = client.post(
        "/handoffs",
        json={
            "conversation_id": "conv-123",
            "verified_customer_id": "",
            "selected_pet_id": "pet_2001",
            "original_concern": "labored breathing",
            "intake_answers": {
                "difficulty_breathing": True,
                "uncontrolled_bleeding": False,
                "collapsed_or_unresponsive": False,
                "known_toxin_exposure": False,
                "rapidly_worsening": False,
            },
        },
    )
    assert response.status_code == 422
