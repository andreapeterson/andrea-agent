import httpx
import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_handoff_client
from app.integrations.handoff_client import HandoffClient
from app.main import app as pawline_app
from mock_services.handoff_api import HANDOFF_STORE, app as mock_handoff_app, reset_handoff_store

TEST_SECRET = "test-handoff-secret"


@pytest.fixture(autouse=True)
def reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HANDOFF_WEBHOOK_SECRET", TEST_SECRET)
    monkeypatch.setenv("MOCK_HANDOFF_WEBHOOK_SECRET", TEST_SECRET)
    reset_handoff_store()
    yield
    reset_handoff_store()


def _urgent_payload() -> dict:
    return {
        "conversation_id": "conv-urgent",
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


def _urgent_with_unanswered_payload() -> dict:
    return {
        "conversation_id": "conv-unanswered",
        "verified_customer_id": "customer_1001",
        "selected_pet_id": "pet_2001",
        "original_concern": "concerned owner",
        "intake_answers": {
            "difficulty_breathing": True,
            "uncontrolled_bleeding": None,
            "collapsed_or_unresponsive": None,
            "known_toxin_exposure": None,
            "rapidly_worsening": None,
        },
    }


def _complete_non_urgent_payload() -> dict:
    return {
        "conversation_id": "conv-routine",
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
        "conversation_id": "conv-uncertain",
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


def _make_real_client() -> HandoffClient:
    return HandoffClient(
        base_url="http://handoff.test",
        webhook_secret=TEST_SECRET,
        transport=httpx.ASGITransport(app=mock_handoff_app),
    )


def test_urgent_handoff_succeeds_end_to_end() -> None:
    pawline_app.dependency_overrides[get_handoff_client] = lambda: _make_real_client()

    try:
        with TestClient(pawline_app) as client:
            response = client.post("/handoffs", json=_urgent_payload())

        assert response.status_code == 202
        payload = response.json()
        assert payload["conversation_id"] == "conv-urgent"
        assert payload["status"] == "queued"
        assert len(HANDOFF_STORE) == 1

        stored = HANDOFF_STORE[0]
        assert stored.conversation_id == "conv-urgent"
        assert stored.routing_decision.routing_level.value == "urgent"
        assert stored.routing_decision.next_action.value == "create_handoff"
        assert stored.summary.original_concern == "labored breathing"
        assert stored.summary.positive_signals == ["difficulty_breathing"]
        assert stored.summary.negative_signals == [
            "uncontrolled_bleeding",
            "collapsed_or_unresponsive",
            "known_toxin_exposure",
            "rapidly_worsening",
        ]
        assert stored.summary.unanswered_signals == []
        assert stored.summary.matched_rule_ids == ["urgent-difficulty-breathing"]
    finally:
        pawline_app.dependency_overrides.clear()


def test_urgent_signal_wins_even_with_unanswered_fields() -> None:
    pawline_app.dependency_overrides[get_handoff_client] = lambda: _make_real_client()

    try:
        with TestClient(pawline_app) as client:
            response = client.post("/handoffs", json=_urgent_with_unanswered_payload())

        assert response.status_code == 202
        assert len(HANDOFF_STORE) == 1
        stored = HANDOFF_STORE[0]
        assert stored.summary.positive_signals == ["difficulty_breathing"]
        assert stored.summary.unanswered_signals == [
            "uncontrolled_bleeding",
            "collapsed_or_unresponsive",
            "known_toxin_exposure",
            "rapidly_worsening",
        ]
        assert stored.summary.negative_signals == []
    finally:
        pawline_app.dependency_overrides.clear()


def test_non_urgent_complete_intake_is_rejected_before_handoff() -> None:
    pawline_app.dependency_overrides[get_handoff_client] = lambda: _make_real_client()

    try:
        with TestClient(pawline_app) as client:
            response = client.post("/handoffs", json=_complete_non_urgent_payload())

        assert response.status_code == 409
        assert response.json()["detail"] == "handoff-not-required"
        assert len(HANDOFF_STORE) == 0
    finally:
        pawline_app.dependency_overrides.clear()


def test_incomplete_intake_without_urgent_signal_is_rejected_before_handoff() -> None:
    pawline_app.dependency_overrides[get_handoff_client] = lambda: _make_real_client()

    try:
        with TestClient(pawline_app) as client:
            response = client.post("/handoffs", json=_incomplete_payload())

        assert response.status_code == 409
        assert response.json()["detail"] == "handoff-not-required"
        assert len(HANDOFF_STORE) == 0
    finally:
        pawline_app.dependency_overrides.clear()


def test_caller_cannot_inject_routing_decision() -> None:
    pawline_app.dependency_overrides[get_handoff_client] = lambda: _make_real_client()

    try:
        with TestClient(pawline_app) as client:
            response = client.post(
                "/handoffs",
                json={
                    **_urgent_payload(),
                    "routing_decision": {
                        "routing_level": "routine",
                        "next_action": "search_routine_appointment",
                        "matched_rule_ids": ["routine-no-escalation-indicators"],
                        "missing_fields": [],
                    },
                },
            )

        assert response.status_code == 422
        assert len(HANDOFF_STORE) == 0
    finally:
        pawline_app.dependency_overrides.clear()


def test_caller_cannot_inject_summary() -> None:
    pawline_app.dependency_overrides[get_handoff_client] = lambda: _make_real_client()

    try:
        with TestClient(pawline_app) as client:
            response = client.post(
                "/handoffs",
                json={
                    **_urgent_payload(),
                    "summary": {
                        "original_concern": "injected",
                        "routing_level": "routine",
                        "positive_signals": [],
                        "negative_signals": [],
                        "unanswered_signals": [],
                        "matched_rule_ids": [],
                    },
                },
            )

        assert response.status_code == 422
        assert len(HANDOFF_STORE) == 0
    finally:
        pawline_app.dependency_overrides.clear()
