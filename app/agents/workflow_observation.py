"""Optional, content-free execution traces; observers never control the workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterator
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
    """Only explicit metadata belongs here, never payloads or exception objects."""

    conversation_id: str
    turn_number: int
    event_type: WorkflowEventType
    specialist: Literal["verification", "policy", "intake", "scheduling", "human_handoff"] | None = None
    tool_name: str | None = None
    prompt_version: str | None = None
    duration_ms: float | None = None
    model: str | None = None
    status: Literal["succeeded", "failed"] | None = None


WorkflowObserver = Callable[[WorkflowEvent], None]


@dataclass(frozen=True)
class _TurnObservation:
    observer: WorkflowObserver
    conversation_id: str
    turn_number: int
    specialist: str | None = None
    prompt_version: str | None = None


_current_turn: ContextVar[_TurnObservation | None] = ContextVar("workflow_observation", default=None)


def _safe_emit(
    event_type: WorkflowEventType, *, tool_name: str | None = None,
    prompt_version: str | None = None, duration_ms: float | None = None,
    model: str | None = None, status: Literal["succeeded", "failed"] | None = None,
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
        ))
    except Exception:
        # Observability is secondary. Do not log the observer's exception or recurse.
        pass


@contextmanager
def _observe_operation(
    started_event: WorkflowEventType, completed_event: WorkflowEventType, *,
    failed_event: WorkflowEventType | None = None, tool_name: str | None = None,
    model: str | None = None, prompt_version: str | None = None,
) -> Iterator[None]:
    if _current_turn.get() is None:
        yield
        return
    started = perf_counter()
    _safe_emit(started_event, tool_name=tool_name, model=model, prompt_version=prompt_version)
    try:
        yield
    except BaseException:
        _safe_emit(
            failed_event or completed_event, tool_name=tool_name, model=model,
            prompt_version=prompt_version, duration_ms=(perf_counter() - started) * 1000,
            status="failed",
        )
        raise
    else:
        _safe_emit(
            completed_event, tool_name=tool_name, model=model, prompt_version=prompt_version,
            duration_ms=(perf_counter() - started) * 1000, status="succeeded",
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


@contextmanager
def observe_model_request(model: str, *, prompt_version: str | None = None) -> Iterator[None]:
    with _observe_operation(
        WorkflowEventType.MODEL_REQUEST_STARTED, WorkflowEventType.MODEL_REQUEST_COMPLETED,
        model=model, prompt_version=prompt_version,
    ):
        yield


@contextmanager
def observe_tool(tool_name: str, *, model: str | None = None) -> Iterator[None]:
    """Observe handler execution and JSON serialization, without inspecting the result."""
    with _observe_operation(
        WorkflowEventType.TOOL_REQUESTED, WorkflowEventType.TOOL_COMPLETED,
        tool_name=tool_name, model=model,
    ):
        yield
