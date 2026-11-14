"""Pure summaries of completed request durations, expressed in milliseconds."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum, isfinite
from numbers import Real


@dataclass(frozen=True)
class LatencySummary:
    request_count: int
    total_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class LatencySpeedup:
    """Responses duration divided by Decisions duration; None means undefined."""

    mean: float | None
    p50: float | None
    p95: float | None


def summarize_latencies(samples: Iterable[Real]) -> LatencySummary:
    """Use nearest-rank percentiles: sorted sample at ceil(p * N), one-indexed.

    Accept only finite, nonnegative real numbers; never discard invalid samples.
    The caller owns measurement and completeness checks, not this calculation.
    """
    durations = []
    for sample in samples:
        if isinstance(sample, bool) or not isinstance(sample, Real):
            raise TypeError("Latency samples must be finite nonnegative real numbers.")
        try:
            duration = float(sample)
        except (OverflowError, ValueError):
            raise ValueError("Latency samples must be finite nonnegative real numbers.") from None
        if not isfinite(duration) or duration < 0:
            raise ValueError("Latency samples must be finite nonnegative real numbers.")
        durations.append(duration)
    if not durations:
        raise ValueError("Latency samples must not be empty.")
    durations.sort()
    count = len(durations)
    try:
        total = fsum(durations)
    except OverflowError:
        raise ValueError("Total latency must be finite.") from None

    def nearest_rank(percent: int) -> float:
        # Integer arithmetic implements ceil(percent * count / 100) exactly.
        return durations[(percent * count + 99) // 100 - 1]

    return LatencySummary(
        request_count=count, total_ms=total, mean_ms=total / count,
        p50_ms=nearest_rank(50), p95_ms=nearest_rank(95),
        minimum_ms=durations[0], maximum_ms=durations[-1],
    )


def calculate_latency_speedup(responses: LatencySummary, decisions: LatencySummary) -> LatencySpeedup:
    """Compare equally sized completed runs; a zero denominator yields N/A."""
    if responses.request_count != decisions.request_count or responses.request_count < 1:
        raise ValueError("Latency comparisons require equal, nonzero request counts.")

    def ratio(numerator: float, denominator: float) -> float | None:
        if denominator == 0:
            return None
        result = numerator / denominator
        return result if isfinite(result) else None

    return LatencySpeedup(
        mean=ratio(responses.mean_ms, decisions.mean_ms),
        p50=ratio(responses.p50_ms, decisions.p50_ms),
        p95=ratio(responses.p95_ms, decisions.p95_ms),
    )


def format_router_latency_report(
    summary: LatencySummary, *, backend: str, model: str, passed: int | None = None, failed: int = 0,
) -> str:
    """Format safe metadata and durations without calling a model or storing data."""
    return "\n".join([
        "Router latency",
        "Scope: local live benchmark",
        f"Backend: {backend}",
        f"Model: {model}",
        f"Requests: {summary.request_count}",
        f"Passed: {summary.request_count if passed is None else passed}",
        f"Failed: {failed}",
        f"Total: {summary.total_ms:.2f} ms",
        f"Mean: {summary.mean_ms:.2f} ms",
        f"p50: {summary.p50_ms:.2f} ms",
        f"p95: {summary.p95_ms:.2f} ms",
        f"Minimum: {summary.minimum_ms:.2f} ms",
        f"Maximum: {summary.maximum_ms:.2f} ms",
    ])


def format_latency_comparison(speedup: LatencySpeedup, *, responses_model: str, decisions_model: str) -> str:
    """Ratios above one favor Decisions; undefined ratios are never called faster."""
    def display(value: float | None) -> str:
        return "N/A (zero duration or unrepresentable ratio)" if value is None else f"{value:.2f}x"

    lines = [
        "Router latency comparison",
        "Scope: local live benchmark; not a universal performance guarantee",
        f"Responses model: {responses_model}",
        f"Decisions model: {decisions_model} (fixed supported model)",
        "Ratio: Responses latency / Decisions latency (>1 favors Decisions)",
        f"Mean-latency speedup: {display(speedup.mean)}",
        f"p50 speedup: {display(speedup.p50)}",
        f"p95 speedup: {display(speedup.p95)}",
    ]
    if responses_model != decisions_model:
        lines.append("Models differ; this is not a same-model comparison.")
    return "\n".join(lines)
