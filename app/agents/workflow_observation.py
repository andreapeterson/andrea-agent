"""Optional, content-free execution traces; observers never control the workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
from typing import Literal


class WorkflowEventType(str, Enum):
    TURN_STARTED = "turn_started"
    SPECIALIST_SELECTED = "specialist_selected"
    MODEL_REQUEST_STARTED = "model_request_started"
    MODEL_REQUEST_COMPLETED = "model_request_completed"
    TOOL_REQUESTED = "tool_requested"
    TOOL_COMPLETED = "tool_completed"
    TURN_COMPLETED = "turn_completed"
    TURN_FAILED = "turn_failed"


@dataclass(frozen=True)
class WorkflowEvent:
    """Only explicit metadata belongs here, never payloads or exception objects.
    represents one thing that happened in a workflow turn, with optional timing and token counts.
    frozen = once created, never modified """

    conversation_id: str
    turn_number: int
    event_type: WorkflowEventType
    specialist: Literal["verification", "policy", "intake", "scheduling", "human_handoff"] | None = None
    tool_name: str | None = None
    prompt_version: str | None = None
    duration_ms: float | None = None
    model: str | None = None
    status: Literal["succeeded", "failed"] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens, self.total_tokens):
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError("Token counts must be nonnegative integers or None")
                if self.event_type != WorkflowEventType.MODEL_REQUEST_COMPLETED:
                    raise ValueError("Token counts belong only to model_request_completed events")


WorkflowObserver = Callable[[WorkflowEvent], None] #any function that takes a WorkflowEvent and returns None


@dataclass(frozen=True)
class _TurnObservation: #temporary information needed while turn is running, shared context to build events
    observer: WorkflowObserver #function that receives finished event
    conversation_id: str
    turn_number: int
    specialist: str | None = None
    prompt_version: str | None = None


_current_turn: ContextVar[_TurnObservation | None] = ContextVar("workflow_observation", default=None) #not a global variable bc it could be overwritten


def _safe_emit( #creates and sends one event
    event_type: WorkflowEventType, *, tool_name: str | None = None,
    prompt_version: str | None = None, duration_ms: float | None = None,
    model: str | None = None, status: Literal["succeeded", "failed"] | None = None,
    input_tokens: int | None = None, output_tokens: int | None = None, total_tokens: int | None = None,
) -> None:
    turn = _current_turn.get()
    if turn is None:
        return
    try:
        turn.observer(WorkflowEvent(
            conversation_id=turn.conversation_id, turn_number=turn.turn_number,
            event_type=event_type, specialist=turn.specialist, tool_name=tool_name,
            prompt_version=prompt_version if prompt_version is not None else turn.prompt_version,
            duration_ms=duration_ms, model=model, status=status,
            input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens,
        ))
    except Exception:
        # Observability is secondary. Do not log the observer's exception or recurse. agent workflow more important than observability
        pass


@contextmanager
def _observe_operation( #core reusable timer 
    started_event: WorkflowEventType, completed_event: WorkflowEventType, *,
    failed_event: WorkflowEventType | None = None, tool_name: str | None = None,
    model: str | None = None, prompt_version: str | None = None,
    token_usage: dict[str, int | None] | None = None,
) -> Iterator[None]:
    if _current_turn.get() is None:
        yield
        return
    started = perf_counter()
    _safe_emit(started_event, tool_name=tool_name, model=model, prompt_version=prompt_version)
    try:
        yield #run actual code inside with block, if it fails, emit failed event, otherwise emit completed event
    except BaseException:
        _safe_emit(
            failed_event or completed_event, tool_name=tool_name, model=model,
            prompt_version=prompt_version, duration_ms=(perf_counter() - started) * 1000,
            status="failed",
            **(token_usage or {}), #Take the key-value pairs inside token_usage and pass them as named arguments. If token_usage is None, use an empty dictionary instead.
        )
        raise
    else:
        _safe_emit(
            completed_event, tool_name=tool_name, model=model, prompt_version=prompt_version,
            duration_ms=(perf_counter() - started) * 1000, status="succeeded",
            **(token_usage or {}),
        )


@contextmanager
def observe_turn(
    observer: WorkflowObserver | None, conversation_id: str, turn_number: int,
) -> Iterator[None]:
    """Bind an optional observer to this async task, restoring it even on failure."""
    turn = _TurnObservation(observer, conversation_id, turn_number) if observer is not None else None
    token = _current_turn.set(turn)
    try:
        with _observe_operation(
            WorkflowEventType.TURN_STARTED, WorkflowEventType.TURN_COMPLETED,
            failed_event=WorkflowEventType.TURN_FAILED,
        ):
            yield
    finally:
        _current_turn.reset(token)


def observe_specialist(specialist: str, *, prompt_version: str | None = None) -> None:
    turn = _current_turn.get()
    if turn is not None:
        _current_turn.set(replace(turn, specialist=specialist, prompt_version=prompt_version))
        _safe_emit(WorkflowEventType.SPECIALIST_SELECTED)


def _usage_field(value: object, name: str) -> object: #Get input_tokens whether usage is a dictionary or an object. Return None if it is missing or broken. parses reponse to get usage field 
    try:
        return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
    except Exception:
        return None


def extract_token_usage(response: object) -> dict[str, int | None]:
    """Copy only three integer counters; never serialize or retain the response."""
    usage = _usage_field(response, "usage")
    counters = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = _usage_field(usage, field)
        counters[field] = value if type(value) is int and value >= 0 else None
    return counters


@contextmanager
def observe_model_request(
    model: str, *, prompt_version: str | None = None,
) -> Iterator[Callable[[object], None]]:
    usage: dict[str, int | None] = {}

    def record_usage(response: object) -> None:
        if _current_turn.get() is not None:
            usage.update(extract_token_usage(response))

    with _observe_operation(
        WorkflowEventType.MODEL_REQUEST_STARTED, WorkflowEventType.MODEL_REQUEST_COMPLETED,
        model=model, prompt_version=prompt_version, token_usage=usage,
    ):
        yield record_usage


@contextmanager
def observe_tool(tool_name: str, *, model: str | None = None) -> Iterator[None]:
    """Observe handler execution and JSON serialization, without inspecting the result."""
    with _observe_operation(
        WorkflowEventType.TOOL_REQUESTED, WorkflowEventType.TOOL_COMPLETED,
        tool_name=tool_name, model=model,
    ):
        yield
