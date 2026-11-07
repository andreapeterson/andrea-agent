"""Trace delivery is optional, isolated, immutable, and never authoritative."""

import asyncio
import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from app.agents import workflow_observation
from app.agents.workflow_observation import (
    WorkflowEvent,
    WorkflowEventType,
    observe_model_request,
    observe_specialist,
    observe_tool,
    observe_turn,
)


def test_event_is_immutable_and_has_no_free_form_payload_fields():
    event = WorkflowEvent("demo-trace", 1, WorkflowEventType.TURN_STARTED)
    assert json.loads(json.dumps(asdict(event)))["event_type"] == "turn_started"
    with pytest.raises(FrozenInstanceError):
        event.specialist = "scheduling"
    with pytest.raises(TypeError):
        WorkflowEvent("demo-trace", 1, WorkflowEventType.TURN_STARTED, tool_arguments={"phone": "private"})


def test_disabled_scope_produces_no_events_even_inside_an_observed_turn():
    events = []
    with observe_turn(events.append, "demo-outer", 1):
        with observe_turn(None, "demo-disabled", 2):
            observe_specialist("policy")
            with observe_model_request("model"), observe_tool("search_policy"):
                pass
    assert [event.event_type for event in events] == [
        WorkflowEventType.TURN_STARTED, WorkflowEventType.TURN_COMPLETED,
    ]
    assert {event.conversation_id for event in events} == {"demo-outer"}


def test_nested_scope_restores_specialist_and_prompt_metadata():
    events = []
    with observe_turn(events.append, "demo-outer", 4):
        observe_specialist("policy", prompt_version="policy-v2")
        with observe_turn(events.append, "demo-inner", 1):
            observe_specialist("verification", prompt_version="verification-v2")
        with observe_tool("search_policy"):
            pass
    tool_events = [event for event in events if event.tool_name is not None]
    assert len(tool_events) == 2
    assert all((event.conversation_id, event.turn_number, event.specialist, event.prompt_version) == (
        "demo-outer", 4, "policy", "policy-v2",
    ) for event in tool_events)


def test_duration_uses_monotonic_elapsed_time_and_never_wall_clock(monkeypatch):
    ticks = iter((10.0, 10.1, 10.35, 10.5))
    monkeypatch.setattr(workflow_observation, "perf_counter", lambda: next(ticks))
    events = []
    with observe_turn(events.append, "demo-timed", 1):
        with observe_model_request("model"):
            pass
    assert events[2].duration_ms == pytest.approx(250.0)
    assert events[-1].duration_ms == pytest.approx(500.0)
    assert events[0].duration_ms is None and events[1].duration_ms is None


def test_observer_return_value_cannot_choose_specialist_or_tool_results():
    events = []

    def observer(event):
        events.append(event)
        return {"specialist": "scheduling", "status": "booking_confirmed"}

    with observe_turn(observer, "demo-return", 1):
        observe_specialist("policy")
        with observe_tool("search_policy"):
            result = {"found": False}
    assert result == {"found": False}
    assert events[-1].specialist == "policy"
    assert events[-1].status == "succeeded"


@pytest.mark.asyncio
async def test_cancellation_records_failure_and_resets_context_without_swallowing_cancellation():
    events = []
    with pytest.raises(asyncio.CancelledError):
        with observe_turn(events.append, "demo-cancelled", 2):
            with observe_model_request("model"):
                raise asyncio.CancelledError()
    assert events[-2].event_type == WorkflowEventType.MODEL_REQUEST_COMPLETED
    assert events[-2].status == "failed"
    assert events[-1].event_type == WorkflowEventType.TURN_FAILED
    assert events[-1].status == "failed"
    before = list(events)
    observe_specialist("policy")
    with observe_tool("search_policy"):
        pass
    assert events == before
