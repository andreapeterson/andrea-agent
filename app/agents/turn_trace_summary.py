"""Pure turn metrics and optional estimates from content-free workflow events."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from app.agents.workflow_observation import WorkflowEvent, WorkflowEventType


@dataclass(frozen=True)
class TurnTraceSummary: #total summary of one finished turn from list of WorkflowEvent objects
    conversation_id: str
    turn_number: int
    selected_specialist: str | None
    prompt_versions: tuple[str, ...]
    model_names: tuple[str, ...]
    model_request_count: int
    tool_names: tuple[str, ...]
    model_duration_ms: float | None
    tool_duration_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    outcome: Literal["completed", "failed"]


def _validate_measurements(event: WorkflowEvent) -> None:
    duration = event.duration_ms
    if duration is not None and (
        type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
    ):
        raise ValueError("Trace durations must be finite nonnegative numbers or None")
    for value in (event.input_tokens, event.output_tokens, event.total_tokens):
        if value is not None:
            if type(value) is not int or value < 0:
                raise ValueError("Trace token counts must be nonnegative integers or None")
            if event.event_type != WorkflowEventType.MODEL_REQUEST_COMPLETED:
                raise ValueError("Token counts belong only to model_request_completed events")


def _complete_total(events: list[WorkflowEvent], field: str) -> int | float | None:
    values = [getattr(event, field) for event in events]
    # Missing values must not become a partial total; no operations means a known zero.
    return sum(values) if all(value is not None for value in values) else None


def summarize_turn_trace(events: Iterable[WorkflowEvent]) -> TurnTraceSummary:
    """Validate one finished serial turn; aggregate only explicit event metadata."""
    events = tuple(events)
    if not events:
        raise ValueError("A turn trace must contain events")
    if not all(isinstance(event, WorkflowEvent) for event in events):
        raise TypeError("A turn trace must contain WorkflowEvent objects")
    if len({event.conversation_id for event in events}) != 1:
        raise ValueError("A turn trace must belong to one conversation")
    if len({event.turn_number for event in events}) != 1:
        raise ValueError("A turn trace must belong to one turn number")
    kinds = [event.event_type for event in events]
    if kinds[0] != WorkflowEventType.TURN_STARTED or kinds.count(WorkflowEventType.TURN_STARTED) != 1:
        raise ValueError("A turn trace must begin with exactly one turn_started event")
    terminal_kinds = {WorkflowEventType.TURN_COMPLETED, WorkflowEventType.TURN_FAILED}
    if kinds[-1] not in terminal_kinds or sum(kind in terminal_kinds for kind in kinds) != 1:
        raise ValueError("A turn trace must end with exactly one turn_completed or turn_failed event")

    selected = None
    model_started = []
    model_completed = []
    tools = []
    tool_completed = []
    pending_model = None
    pending_tool = None
    for event in events:
        _validate_measurements(event)
        kind = event.event_type
        if kind == WorkflowEventType.SPECIALIST_SELECTED:
            if selected is not None or not event.specialist:
                raise ValueError("A turn trace may select one known specialist at most once")
            selected = event.specialist
        elif kind == WorkflowEventType.MODEL_REQUEST_STARTED:
            if pending_model is not None:
                raise ValueError("A serial turn cannot have overlapping model requests")
            pending_model = event
            model_started.append(event)
        elif kind == WorkflowEventType.MODEL_REQUEST_COMPLETED:
            if pending_model is None or pending_model.model != event.model:
                raise ValueError("model_request_completed requires a matching earlier model_request_started")
            pending_model = None
            model_completed.append(event)
        elif kind == WorkflowEventType.TOOL_REQUESTED:
            if pending_tool is not None or not event.tool_name:
                raise ValueError("Tool requests must be named and execute in serial order")
            pending_tool = event
            tools.append(event.tool_name)
        elif kind == WorkflowEventType.TOOL_COMPLETED:
            if pending_tool is None or pending_tool.tool_name != event.tool_name:
                raise ValueError("tool_completed requires a matching earlier tool_requested")
            pending_tool = None
            tool_completed.append(event)
        elif kind not in terminal_kinds and kind != WorkflowEventType.TURN_STARTED:
            raise ValueError("Unknown workflow event type")
    if pending_model is not None or pending_tool is not None:
        raise ValueError("A finished turn cannot contain unfinished model or tool operations")
    return TurnTraceSummary(
        conversation_id=events[0].conversation_id, turn_number=events[0].turn_number,
        selected_specialist=selected,
        prompt_versions=tuple(dict.fromkeys(event.prompt_version for event in model_started if event.prompt_version is not None)),
        model_names=tuple(dict.fromkeys(event.model for event in model_started if event.model is not None)),
        model_request_count=len(model_started), tool_names=tuple(tools),
        model_duration_ms=_complete_total(model_completed, "duration_ms"),
        tool_duration_ms=_complete_total(tool_completed, "duration_ms"),
        input_tokens=_complete_total(model_completed, "input_tokens"),
        output_tokens=_complete_total(model_completed, "output_tokens"),
        total_tokens=_complete_total(model_completed, "total_tokens"),
        outcome="completed" if kinds[-1] == WorkflowEventType.TURN_COMPLETED else "failed",
    )


@dataclass(frozen=True)
class ModelTokenPricing:
    """Injected USD prices per 1,000,000 input or output tokens; never live defaults."""

    input_per_million: Decimal
    output_per_million: Decimal

    def __post_init__(self) -> None:
        for rate in (self.input_per_million, self.output_per_million):
            if not isinstance(rate, Decimal) or not rate.is_finite() or rate < 0:
                raise ValueError("Token prices must be finite nonnegative Decimal USD rates per million tokens")


def estimate_turn_cost(
    events: Iterable[WorkflowEvent], pricing: Mapping[str, ModelTokenPricing],
) -> Decimal | None:
    """Estimate USD using each request's injected rates; incomplete coverage returns None.

    This is a basic token-cost estimate, not a bill: discounts, caching tiers,
    and other charges are outside this two-rate mapping. It never fetches prices.
    """
    events = tuple(events)
    summarize_turn_trace(events)
    estimate = Decimal(0)
    for event in events:
        if event.event_type != WorkflowEventType.MODEL_REQUEST_COMPLETED:
            continue
        rates = pricing.get(event.model) if event.model is not None else None
        if rates is None or any(value is None for value in (event.input_tokens, event.output_tokens, event.total_tokens)):
            return None
        estimate += (
            Decimal(event.input_tokens) * rates.input_per_million
            + Decimal(event.output_tokens) * rates.output_per_million
        ) / Decimal(1_000_000)
    return estimate


def format_turn_trace_summary(summary: TurnTraceSummary, *, estimated_cost: Decimal | None = None) -> str:
    """Format only safe summary fields; no event payload or raw response is accepted."""
    def count(value: int | None) -> str:
        return "unknown" if value is None else str(value)

    def duration(value: float | None) -> str:
        return "unknown" if value is None else f"{value:.2f} ms"

    lines = [
        f"Conversation: {summary.conversation_id}", f"Turn: {summary.turn_number}",
        f"Outcome: {summary.outcome}", f"Specialist: {summary.selected_specialist or 'unknown'}",
        f"Models: {', '.join(summary.model_names) or 'none'}",
        f"Prompt versions: {', '.join(summary.prompt_versions) or 'none'}",
        f"Model requests: {summary.model_request_count}",
        f"Tools (execution order): {', '.join(summary.tool_names) or 'none'}",
        f"Model duration: {duration(summary.model_duration_ms)}",
        f"Tool duration: {duration(summary.tool_duration_ms)}",
        f"Input tokens: {count(summary.input_tokens)}", f"Output tokens: {count(summary.output_tokens)}",
        f"Total tokens: {count(summary.total_tokens)}",
        "Estimated cost (USD): " + (format(estimated_cost, "f") if estimated_cost is not None else "unavailable"),
    ]
    return "\n".join(lines)
