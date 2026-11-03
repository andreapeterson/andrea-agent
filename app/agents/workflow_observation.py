"""Optional, turn-scoped route and tool observations without conversation content."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class WorkflowEvent:
    conversation_id: str
    turn_number: int
    specialist: str
    tool_name: str | None = None


WorkflowObserver = Callable[[WorkflowEvent], None]


@dataclass(frozen=True)
class _TurnObservation:
    observer: WorkflowObserver
    conversation_id: str
    turn_number: int
    specialist: str | None = None


_current_turn: ContextVar[_TurnObservation | None] = ContextVar("workflow_observation", default=None) #keeps


@contextmanager
def observe_turn(
    observer: WorkflowObserver | None, conversation_id: str, turn_number: int,
) -> Iterator[None]:
    """Bind an optional observer to this async task, restoring it even on failure."""
    turn = _TurnObservation(observer, conversation_id, turn_number) if observer is not None else None
    token = _current_turn.set(turn)
    try:
        yield
    finally:
        _current_turn.reset(token)


def observe_specialist(specialist: str) -> None:
    turn = _current_turn.get()
    if turn is not None:
        _current_turn.set(replace(turn, specialist=specialist))
        turn.observer(WorkflowEvent(turn.conversation_id, turn.turn_number, specialist))


def observe_tool(tool_name: str) -> None:
    turn = _current_turn.get()
    if turn is not None:
        if turn.specialist is None:
            raise RuntimeError("Tool observation requires a routed specialist")
        turn.observer(WorkflowEvent(turn.conversation_id, turn.turn_number, turn.specialist, tool_name))
