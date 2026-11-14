"""Trace delivery is optional, isolated, immutable, and never authoritative."""

import asyncio
import json
import weakref
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import pytest

from app.agents import workflow_observation
from app.agents.workflow_observation import (
    WorkflowEvent,
    WorkflowEventType,
    extract_token_usage,
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
    with observe_turn(events.append, "demo-outer", 1), observe_turn(None, "demo-disabled", 2):
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
    with observe_turn(events.append, "demo-timed", 1), observe_model_request("model"):
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
    with pytest.raises(asyncio.CancelledError), observe_turn(events.append, "demo-cancelled", 2), observe_model_request("model"):
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


@pytest.mark.parametrize("bad", [-1, True, False, 1.5, "12", float("nan"), [], {}])
def test_malformed_usage_values_are_ignored_without_coercion(bad):
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=bad, output_tokens=0, total_tokens=2))
    assert extract_token_usage(response) == {"input_tokens": None, "output_tokens": 0, "total_tokens": 2}
    with pytest.raises(ValueError, match="nonnegative integers"):
        WorkflowEvent("demo", 1, WorkflowEventType.MODEL_REQUEST_COMPLETED, input_tokens=bad)


@pytest.mark.parametrize("response", [None, object(), {}, {"usage": None}, {"usage": []}])
def test_missing_usage_produces_unknown_counts(response):
    assert extract_token_usage(response) == {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def test_usage_access_failures_are_ignored_and_nested_details_are_never_read():
    class BrokenResponse:
        @property
        def usage(self):
            raise RuntimeError("PRIVATE_API_RESPONSE_EXCEPTION")

    assert all(value is None for value in extract_token_usage(BrokenResponse()).values())

    class Usage:
        input_tokens = 4
        output_tokens = 2

        @property
        def total_tokens(self):
            raise RuntimeError("PRIVATE_USAGE_EXCEPTION")

        @property
        def input_tokens_details(self):
            raise AssertionError("Do not read cached prompt details")

        @property
        def output_tokens_details(self):
            raise AssertionError("Do not read reasoning details")

    assert extract_token_usage(SimpleNamespace(usage=Usage())) == {
        "input_tokens": 4, "output_tokens": 2, "total_tokens": None,
    }


def test_capture_keeps_only_integer_counts_not_the_response():
    class Response:
        def __init__(self):
            self.usage = {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}

        def model_dump(self):
            raise AssertionError("Do not serialize the raw OpenAI response")

    events = []
    response = Response()
    reference = weakref.ref(response)
    with observe_turn(events.append, "demo-usage", 1), observe_model_request("model") as record_usage:
        record_usage(response)
    del response
    assert reference() is None
    completed = events[2]
    assert completed.event_type == WorkflowEventType.MODEL_REQUEST_COMPLETED
    assert (completed.input_tokens, completed.output_tokens, completed.total_tokens) == (4, 2, 6)
    assert all(event.input_tokens is None for event in events if event is not completed)


def test_disabled_observer_does_not_inspect_usage():
    class Response:
        @property
        def usage(self):
            pytest.fail("No usage inspection is needed when observation is disabled")

    with observe_turn(None, "demo-disabled", 1), observe_model_request("model") as record_usage:
        record_usage(Response())


def test_token_counts_cannot_be_attached_to_non_model_events():
    with pytest.raises(ValueError, match="only to model_request_completed"):
        WorkflowEvent("demo", 1, WorkflowEventType.TOOL_COMPLETED, input_tokens=1)
