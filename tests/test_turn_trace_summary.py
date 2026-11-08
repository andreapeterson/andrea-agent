"""Test pure trace aggregation and estimates using explicitly fictional rates."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from app.agents.turn_trace_summary import (
    ModelTokenPricing,
    TurnTraceSummary,
    estimate_turn_cost,
    format_turn_trace_summary,
    summarize_turn_trace,
)
from app.agents.workflow_observation import WorkflowEvent, WorkflowEventType as Kind


def event(kind, **values):
    return WorkflowEvent("demo-summary", 2, kind, **values)


@pytest.fixture
def trace():
    return [
        event(Kind.TURN_STARTED),
        event(Kind.MODEL_REQUEST_STARTED, model="demo-router", prompt_version="router-v6"),
        event(Kind.MODEL_REQUEST_COMPLETED, model="demo-router", duration_ms=4.0,
              input_tokens=30, output_tokens=5, total_tokens=35, status="succeeded"),
        event(Kind.SPECIALIST_SELECTED, specialist="policy", prompt_version="policy-v2"),
        event(Kind.MODEL_REQUEST_STARTED, model="demo-specialist", prompt_version="policy-v2"),
        event(Kind.MODEL_REQUEST_COMPLETED, model="demo-specialist", duration_ms=10.0,
              input_tokens=20, output_tokens=10, total_tokens=30, status="succeeded"),
        event(Kind.TOOL_REQUESTED, tool_name="search_policy"),
        event(Kind.TOOL_COMPLETED, tool_name="search_policy", duration_ms=2.0, status="succeeded"),
        event(Kind.MODEL_REQUEST_STARTED, model="demo-specialist", prompt_version="policy-v2"),
        event(Kind.MODEL_REQUEST_COMPLETED, model="demo-specialist", duration_ms=6.0,
              input_tokens=25, output_tokens=5, total_tokens=30, status="succeeded"),
        event(Kind.TURN_COMPLETED, duration_ms=25.0, status="succeeded"),
    ]


@pytest.fixture
def rates():
    # Arbitrary fixture values, not advertised prices for any OpenAI model.
    return {
        "demo-router": ModelTokenPricing(Decimal("2"), Decimal("4")),
        "demo-specialist": ModelTokenPricing(Decimal("1"), Decimal("3")),
    }


def test_summary_aggregates_all_requests_without_double_counting_turn_latency(trace):
    before = deepcopy(trace)
    summary = summarize_turn_trace(iter(trace))
    assert summary == TurnTraceSummary(
        conversation_id="demo-summary", turn_number=2, selected_specialist="policy",
        prompt_versions=("router-v6", "policy-v2"), model_names=("demo-router", "demo-specialist"),
        model_request_count=3, tool_names=("search_policy",), model_duration_ms=20.0,
        tool_duration_ms=2.0, input_tokens=75, output_tokens=20, total_tokens=95, outcome="completed",
    )
    assert trace == before
    with pytest.raises(FrozenInstanceError):
        summary.total_tokens = 0
    assert isinstance(summary.tool_names, tuple)


def test_tool_execution_order_and_repeated_tools_are_preserved(trace):
    trace[6:8] = [
        event(Kind.TOOL_REQUESTED, tool_name="record_intake"),
        event(Kind.TOOL_COMPLETED, tool_name="record_intake", duration_ms=2.0),
        event(Kind.TOOL_REQUESTED, tool_name="request_handoff"),
        event(Kind.TOOL_COMPLETED, tool_name="request_handoff", duration_ms=3.0),
        event(Kind.TOOL_REQUESTED, tool_name="record_intake"),
        event(Kind.TOOL_COMPLETED, tool_name="record_intake", duration_ms=1.0),
    ]
    summary = summarize_turn_trace(trace)
    assert summary.tool_names == ("record_intake", "request_handoff", "record_intake")
    assert summary.tool_duration_ms == 6.0


def test_failed_turn_preserves_usage_from_earlier_requests_and_unknown_failed_usage(trace):
    trace[-2] = replace(trace[-2], status="failed", input_tokens=None, output_tokens=None, total_tokens=None)
    trace[-1] = event(Kind.TURN_FAILED, status="failed", duration_ms=25.0)
    summary = summarize_turn_trace(trace)
    assert summary.outcome == "failed" and summary.model_request_count == 3
    assert summary.input_tokens is summary.output_tokens is summary.total_tokens is None
    assert summary.model_duration_ms == 20.0
    assert "Outcome: failed" in format_turn_trace_summary(summary)


def test_completed_trace_with_no_requests_has_known_zero_counts_and_no_pricing_requirement():
    trace = [event(Kind.TURN_STARTED), event(Kind.TURN_COMPLETED)]
    summary = summarize_turn_trace(trace)
    assert summary.selected_specialist is None and summary.model_names == summary.prompt_versions == ()
    assert summary.model_request_count == summary.input_tokens == summary.output_tokens == summary.total_tokens == 0
    assert summary.model_duration_ms == summary.tool_duration_ms == 0
    assert estimate_turn_cost(trace, {}) == Decimal(0)


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "total_tokens"])
def test_missing_usage_is_unknown_per_counter_and_never_estimated(trace, rates, field):
    trace[2] = replace(trace[2], **{field: None})
    summary = summarize_turn_trace(trace)
    assert getattr(summary, field) is None
    assert estimate_turn_cost(trace, rates) is None
    for other, total in (("input_tokens", 75), ("output_tokens", 20), ("total_tokens", 95)):
        if other != field:
            assert getattr(summary, other) == total
    assert "unknown" in format_turn_trace_summary(summary)


def test_missing_duration_is_unknown_instead_of_a_partial_latency_sum(trace):
    trace[2] = replace(trace[2], duration_ms=None)
    summary = summarize_turn_trace(trace)
    assert summary.model_duration_ms is None
    assert summary.tool_duration_ms == 2.0
    assert "Model duration: unknown" in format_turn_trace_summary(summary)


@pytest.mark.parametrize("change, message", [
    ("conversation", "one conversation"), ("turn", "one turn number"),
    ("no_start", "turn_started"), ("duplicate_start", "turn_started"),
    ("both_terminals", "exactly one turn_completed or turn_failed"),
    ("no_terminal", "exactly one turn_completed or turn_failed"),
    ("no_tool_request", "tool_completed requires"), ("wrong_tool_completion", "tool_completed requires"),
    ("no_model_request", "model_request_completed requires"),
    ("wrong_model_completion", "model_request_completed requires"),
    ("unfinished_model", "unfinished model or tool"), ("unfinished_tool", "unfinished model or tool"),
    ("overlapping_model", "overlapping model"), ("overlapping_tool", "serial order"),
    ("duplicate_selection", "select one known specialist"), ("unknown_kind", "Unknown workflow event"),
])
def test_structurally_invalid_traces_are_rejected(trace, change, message):
    if change == "conversation":
        trace[-1] = replace(trace[-1], conversation_id="demo-other")
    elif change == "turn":
        trace[-1] = replace(trace[-1], turn_number=3)
    elif change == "no_start":
        trace.pop(0)
    elif change == "duplicate_start":
        trace.insert(1, event(Kind.TURN_STARTED))
    elif change == "both_terminals":
        trace.append(event(Kind.TURN_FAILED))
    elif change == "no_terminal":
        trace.pop()
    elif change == "no_tool_request":
        trace.pop(6)
    elif change == "wrong_tool_completion":
        trace[7] = replace(trace[7], tool_name="request_handoff")
    elif change == "no_model_request":
        trace.pop(1)
    elif change == "wrong_model_completion":
        trace[2] = replace(trace[2], model="demo-other")
    elif change == "unfinished_model":
        trace.pop(-2)
    elif change == "unfinished_tool":
        trace.pop(7)
    elif change == "overlapping_model":
        trace.insert(2, trace[1])
    elif change == "overlapping_tool":
        trace.insert(7, trace[6])
    elif change == "duplicate_selection":
        trace.insert(4, trace[3])
    else:
        trace.insert(1, event("unsupported"))
    with pytest.raises(ValueError, match=message):
        summarize_turn_trace(trace)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "2"])
def test_negative_nonfinite_or_malformed_durations_are_rejected(trace, value):
    trace[2] = replace(trace[2], duration_ms=value)
    with pytest.raises(ValueError, match="finite nonnegative"):
        summarize_turn_trace(trace)


def test_summary_defensively_rejects_invalid_token_counts_even_if_event_validation_was_bypassed(trace):
    object.__setattr__(trace[2], "input_tokens", -1)
    with pytest.raises(ValueError, match="nonnegative integers"):
        summarize_turn_trace(trace)


def test_empty_or_untyped_traces_are_rejected():
    with pytest.raises(ValueError, match="must contain events"):
        summarize_turn_trace([])
    with pytest.raises(TypeError, match="WorkflowEvent"):
        summarize_turn_trace([{}])


def test_estimate_uses_each_model_rate_and_decimal_units(trace, rates):
    # (30*2 + 5*4 + 45*1 + 15*3) / 1,000,000 USD.
    assert estimate_turn_cost(iter(trace), rates) == Decimal("0.00017")
    assert estimate_turn_cost(trace, rates) == (
        Decimal(30) * Decimal(2) + Decimal(5) * Decimal(4)
        + Decimal(45) * Decimal(1) + Decimal(15) * Decimal(3)
    ) / Decimal(1_000_000)


def test_unknown_pricing_or_unknown_model_returns_none(trace, rates):
    assert estimate_turn_cost(trace, {}) is None
    assert estimate_turn_cost(trace, {"demo-router": rates["demo-router"]}) is None
    trace[1] = replace(trace[1], model=None)
    trace[2] = replace(trace[2], model=None)
    assert estimate_turn_cost(trace, rates) is None


@pytest.mark.parametrize("rate", [Decimal("-1"), Decimal("NaN"), Decimal("Infinity"), 2.0, "2"])
def test_prices_require_nonnegative_finite_decimal_values(rate):
    with pytest.raises(ValueError, match="Decimal USD rates per million"):
        ModelTokenPricing(rate, Decimal(0))


def test_terminal_report_displays_only_summary_metadata_and_labels_estimates(trace, rates):
    summary = summarize_turn_trace(trace)
    report = format_turn_trace_summary(summary, estimated_cost=estimate_turn_cost(trace, rates))
    assert "Specialist: policy" in report
    assert "Models: demo-router, demo-specialist" in report
    assert "Prompt versions: router-v6, policy-v2" in report
    assert "Model requests: 3" in report
    assert "Tools (execution order): search_policy" in report
    assert "Model duration: 20.00 ms" in report and "Tool duration: 2.00 ms" in report
    assert "Input tokens: 75" in report and "Output tokens: 20" in report and "Total tokens: 95" in report
    assert "Estimated cost (USD): 0.00017" in report
    assert "Outcome: completed" in report
    assert "Estimated cost (USD): unavailable" in format_turn_trace_summary(summary)
    for forbidden in ("tool_arguments", "tool_results", "api_key", "authorization", "traceback", "output_text"):
        assert forbidden not in report
