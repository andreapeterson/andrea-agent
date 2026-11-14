"""Deterministic latency statistics and Router benchmark boundaries; no live API."""

import asyncio
import socket
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai import AsyncOpenAI

from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from evals import live_router_eval
from evals.latency import (
    calculate_latency_speedup,
    format_latency_comparison,
    summarize_latencies,
)
from evals.router_eval import load_router_cases, run_router_evaluation

SMOKE_CASE = "policy_clinic_hours_unverified"


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Latency tests must not construct API clients or make network requests.")

    monkeypatch.setattr(AsyncOpenAI, "__init__", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_API_KEY")
    monkeypatch.delenv("OPENAI_ROUTER_MODEL", raising=False)


@pytest.mark.parametrize(("samples", "p50", "p95"), [
    ([100, 20, 60, 40, 80], 60, 100),
    (list(range(1, 21)), 10, 19),
    (list(range(1, 28)), 14, 26),
    ([10, 20], 10, 20),
])
def test_summary_uses_nearest_rank_percentiles_without_changing_samples(samples, p50, p95):
    before = list(samples)

    summary = summarize_latencies(samples)

    assert summary.request_count == len(samples)
    assert summary.total_ms == sum(samples)
    assert summary.mean_ms == sum(samples) / len(samples)
    assert summary.p50_ms == p50
    assert summary.p95_ms == p95
    assert summary.minimum_ms == min(samples)
    assert summary.maximum_ms == max(samples)
    assert samples == before


@pytest.mark.parametrize("sample", [0, 12.5])
def test_one_sample_is_supported(sample):
    summary = summarize_latencies([sample])
    assert summary.request_count == 1
    assert all(value == sample for name, value in asdict(summary).items() if name != "request_count")


def test_summary_accepts_an_iterator():
    assert summarize_latencies(iter([0.5, 1.5])).total_ms == 2.0


@pytest.mark.parametrize("samples", [[], (), iter(())])
def test_empty_samples_are_rejected(samples):
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_latencies(samples)


@pytest.mark.parametrize("invalid", [
    True, False, -0.1, float("nan"), float("inf"), -float("inf"),
    "PRIVATE_SAMPLE", "1.0", None, {}, [], 1 + 2j, 10 ** 1000,
], ids=["true", "false", "negative", "nan", "inf", "negative-inf", "private-text", "numeric-text", "none", "dict", "list", "complex", "overflow"])
def test_invalid_samples_are_not_discarded_or_echoed(invalid):
    with pytest.raises((TypeError, ValueError), match="finite nonnegative real numbers") as error:
        summarize_latencies([10, invalid, 20])
    assert "PRIVATE_SAMPLE" not in str(error.value)


def test_overflowing_total_is_rejected():
    with pytest.raises(ValueError, match="Total latency must be finite"):
        summarize_latencies([1e308, 1e308])


class FakeClock:
    def __init__(self):
        self.seconds = 0.0
        self.events = []

    def advance(self, seconds):
        self.seconds += seconds

    def __call__(self):
        self.events.append("clock")
        return self.seconds


@pytest.mark.asyncio
async def test_timer_surrounds_only_the_awaited_route_call(monkeypatch):
    clock = FakeClock()
    client = object()
    context = PromptContext()
    samples = []
    renderer = Mock(spec=PromptRenderer)

    def render(actual_context):
        assert actual_context is context
        clock.events.append("render")
        clock.advance(10)
        return "PRIVATE_RENDERED_PROMPT"

    class Result:
        @property
        def destination(self):
            clock.events.append("destination")
            clock.advance(20)
            return "policy"

    async def route(**kwargs):
        assert kwargs == {"instructions": "PRIVATE_RENDERED_PROMPT", "user_message": "PRIVATE_CALLER"}
        clock.events.append("route_start")
        clock.advance(0.125)
        clock.events.append("route_end")
        return Result()

    def create(*args, **kwargs):
        assert args == (client,)
        clock.events.append("construct")
        clock.advance(30)
        return SimpleNamespace(route=route)

    renderer.render_router.side_effect = render
    monkeypatch.setattr(live_router_eval, "create_router", create)
    run_case = live_router_eval.create_live_route_case(
        client, "gpt-6-luna", renderer, duration_samples=samples, clock=clock,
    )

    assert await run_case(context, "PRIVATE_CALLER") == "policy"

    assert samples == [125.0]
    assert clock.events == ["construct", "render", "clock", "route_start", "route_end", "clock", "destination"]
    assert all(type(sample) is float for sample in samples)


@pytest.mark.asyncio
async def test_each_successful_call_records_once_but_failed_calls_do_not(monkeypatch):
    clock = FakeClock()
    attempts = 0

    async def route(**kwargs):
        nonlocal attempts
        attempts += 1
        clock.advance(0.25)
        if attempts == 3:
            raise RuntimeError("PRIVATE_UPSTREAM_RESPONSE")
        return SimpleNamespace(destination="policy")

    monkeypatch.setattr(live_router_eval, "create_router", lambda *args, **kwargs: SimpleNamespace(route=route))
    samples = []
    run_case = live_router_eval.create_live_route_case(
        object(), "gpt-6-luna", PromptRenderer(), duration_samples=samples, clock=clock,
    )
    for _ in range(2):
        assert await run_case(PromptContext(), "Hours?") == "policy"
    with pytest.raises(RuntimeError):
        await run_case(PromptContext(), "Hours?")

    assert samples == [250.0, 250.0]
    assert len(clock.events) == 5  # Start/end for each success; only start for failure.


class FakeClient:
    def __init__(self, clock):
        self.clock = clock
        self.closed = False

    async def __aenter__(self):
        self.clock.advance(10)
        return self

    async def __aexit__(self, *args):
        self.clock.advance(10)
        self.closed = True


@pytest.fixture
def cli_setup(monkeypatch):
    clock = FakeClock()
    client = FakeClient(clock)
    router = SimpleNamespace(route=AsyncMock())

    async def route(**kwargs):
        clock.advance(0.125)
        return SimpleNamespace(destination="policy")

    router.route.side_effect = route
    factory = Mock(return_value=router)
    monkeypatch.setattr(live_router_eval, "AsyncOpenAI", Mock(return_value=client))
    monkeypatch.setattr(live_router_eval, "create_router", factory)
    monkeypatch.setattr(live_router_eval, "perf_counter", clock)
    return SimpleNamespace(clock=clock, client=client, factory=factory, router=router)


@pytest.mark.parametrize("backend", ["responses", "decisions"])
def test_cli_prints_per_request_milliseconds_after_accuracy_excluding_setup_and_formatting(monkeypatch, capsys, cli_setup, backend):
    original_loader = live_router_eval.load_router_cases
    original_formatter = live_router_eval.format_router_report

    def load():
        cli_setup.clock.advance(30)
        return original_loader()

    def format_accuracy(report):
        cli_setup.clock.advance(40)
        return original_formatter(report)

    monkeypatch.setattr(live_router_eval, "load_router_cases", load)
    monkeypatch.setattr(live_router_eval, "format_router_report", format_accuracy)

    status = live_router_eval.main(["--confirm-live", "--router-backend", backend, "--case-id", SMOKE_CASE])

    assert status == 0
    assert cli_setup.client.closed
    cli_setup.factory.assert_called_once_with(cli_setup.client, backend=backend, model="gpt-6-luna")
    cli_setup.router.route.assert_awaited_once()
    output = capsys.readouterr()
    assert "Accuracy: 100.0%" in output.out and output.err == ""
    assert output.out.endswith(
        f"\nRouter latency\nScope: local live benchmark\nBackend: {backend}\nModel: gpt-6-luna\nRequests: 1\n"
        "Passed: 1\nFailed: 0\n"
        "Total: 125.00 ms\nMean: 125.00 ms\np50: 125.00 ms\np95: 125.00 ms\n"
        "Minimum: 125.00 ms\nMaximum: 125.00 ms\n"
    )
    assert output.out.index("Router evaluation") < output.out.index("Router latency")
    assert "PRIVATE" not in output.out


@pytest.mark.asyncio
async def test_timing_does_not_change_accuracy_report_or_scoring(monkeypatch, cli_setup):
    cases = load_router_cases()[:3]
    expected = await run_router_evaluation(cases, AsyncMock(side_effect=[case.expected_destination for case in cases]))
    destinations = iter(case.expected_destination for case in cases)

    async def route(**kwargs):
        cli_setup.clock.advance(0.125)
        return SimpleNamespace(destination=next(destinations))

    cli_setup.router.route.side_effect = route
    samples = []
    actual = await live_router_eval._evaluate(cases, "PRIVATE_API_KEY", "gpt-6-luna", "responses", samples)

    assert actual == expected
    assert samples == [125.0] * len(cases)
    assert "latency" not in actual and "duration_samples" not in actual


def test_accuracy_failure_suppresses_completed_latency_even_with_a_full_sample(capsys, cli_setup):
    cli_setup.router.route.side_effect = None
    cli_setup.router.route.return_value = SimpleNamespace(destination="verification")

    assert live_router_eval.main(["--confirm-live", "--case-id", SMOKE_CASE]) == 1

    output = capsys.readouterr()
    assert "Failed: 1" in output.out and "router_selection: 1" in output.out
    assert "Router latency" not in output.out and output.err == ""


@pytest.mark.parametrize("samples", [[], [10, 20], [True], [-1], [float("nan")], [float("inf")], ["PRIVATE_SAMPLE"]])
def test_incomplete_or_invalid_timing_cannot_be_presented_as_a_completed_benchmark(monkeypatch, capsys, samples):
    case = next(case for case in load_router_cases() if case.case_id == SMOKE_CASE)
    report = asyncio.run(run_router_evaluation([case], AsyncMock(return_value="policy")))
    before = deepcopy(report)

    async def evaluate(cases, api_key, model, backend, duration_samples):
        duration_samples.extend(samples)
        return report

    monkeypatch.setattr(live_router_eval, "_evaluate", evaluate)
    assert live_router_eval.main(["--confirm-live", "--case-id", SMOKE_CASE]) == 1

    assert report == before
    output = capsys.readouterr()
    assert "Accuracy: 100.0%" in output.out
    assert "Router latency" not in output.out
    assert output.err == "Router latency benchmark incomplete; no latency summary was produced.\n"
    assert "PRIVATE" not in output.out + output.err


def test_partial_live_failure_exposes_no_timing_success_prompts_or_upstream_response(monkeypatch, capsys):
    async def evaluate(cases, api_key, model, backend, duration_samples):
        duration_samples.append(100.0)
        raise RuntimeError("PRIVATE_API_KEY PRIVATE_CALLER PRIVATE_PROMPT PRIVATE_UPSTREAM_RESPONSE")

    monkeypatch.setattr(live_router_eval, "_evaluate", evaluate)
    assert live_router_eval.main(["--confirm-live", "--all-cases"]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Live Router evaluation failed; no report was produced.\n"


def test_invalid_clock_sample_remains_a_safe_local_failure(capsys, cli_setup, monkeypatch):
    clock = Mock(side_effect=[1.0, 0.0])
    monkeypatch.setattr(live_router_eval, "perf_counter", clock)

    assert live_router_eval.main(["--confirm-live", "--case-id", SMOKE_CASE]) == 1

    output = capsys.readouterr()
    assert "Router latency" not in output.out
    assert output.err == "Router latency benchmark incomplete; no latency summary was produced.\n"


def test_speedups_compare_each_statistic_separately():
    responses = summarize_latencies([20, 40, 100])
    decisions = summarize_latencies([10, 20, 20])

    speedup = calculate_latency_speedup(responses, decisions)

    assert speedup.mean == pytest.approx(3.2)
    assert speedup.p50 == 2
    assert speedup.p95 == 5
    text = format_latency_comparison(speedup, responses_model="same-model", decisions_model="same-model")
    assert "Mean-latency speedup: 3.20x" in text
    assert "p50 speedup: 2.00x" in text and "p95 speedup: 5.00x" in text
    assert "Responses latency / Decisions latency" in text
    assert "not a universal performance guarantee" in text


def test_speedup_with_zero_denominators_is_explicitly_unavailable():
    speedup = calculate_latency_speedup(summarize_latencies([10, 20]), summarize_latencies([0, 0]))
    assert asdict(speedup) == {"mean": None, "p50": None, "p95": None}
    text = format_latency_comparison(speedup, responses_model="same-model", decisions_model="same-model")
    assert text.count("N/A (zero duration or unrepresentable ratio)") == 3
    assert "inf" not in text and "nan" not in text


def test_speedup_rejects_mismatched_request_counts():
    with pytest.raises(ValueError, match="equal, nonzero request counts"):
        calculate_latency_speedup(summarize_latencies([1]), summarize_latencies([1, 2]))


def test_both_backends_use_identical_case_order_separately_and_sequentially(monkeypatch, capsys, cli_setup):
    selected_ids = {SMOKE_CASE, "verification_injection_pretend_verified"}
    selected = [case for case in load_router_cases() if case.case_id in selected_ids]
    destinations = {case.user_message: case.expected_destination for case in selected}
    events = []
    samples_by_backend = {}
    selections = []
    original_evaluate = live_router_eval._evaluate

    async def evaluate(cases, key, model, backend, samples):
        selections.append(cases)
        events.append((backend, "start"))
        report = await original_evaluate(cases, key, model, backend, samples)
        events.append((backend, "end"))
        samples_by_backend[backend] = list(samples)
        return report

    def create(client, *, backend, model):
        assert model == "gpt-6-luna"
        durations = iter([0.1, 0.2] if backend == "responses" else [0.025, 0.025])

        async def route(*, instructions, user_message):
            events.append((backend, user_message))
            cli_setup.clock.advance(next(durations))
            return SimpleNamespace(destination=destinations[user_message])

        return SimpleNamespace(route=route)

    monkeypatch.setattr(live_router_eval, "_evaluate", evaluate)
    monkeypatch.setattr(live_router_eval, "create_router", create)

    status = live_router_eval.main([
        "--confirm-live", "--router-backend", "both", "--case-id", SMOKE_CASE,
        "--case-id", "verification_injection_pretend_verified", "--case-id", SMOKE_CASE,
    ])

    assert status == 0
    assert selections[0] is selections[1]
    assert [case.case_id for case in selections[0]] == [case.case_id for case in selected]
    assert events == [
        (backend, event)
        for backend in ("responses", "decisions")
        for event in ["start", *[case.user_message for case in selected], "end"]
    ]
    assert samples_by_backend["responses"] == pytest.approx([100, 200])
    assert samples_by_backend["decisions"] == pytest.approx([25, 25])
    assert all(isinstance(sample, float) for samples in samples_by_backend.values() for sample in samples)
    output = capsys.readouterr()
    responses_text, decisions_text = output.out.split("Router backend: decisions; model: gpt-6-luna", 1)
    assert "Requests: 2\nPassed: 2\nFailed: 0" in responses_text
    assert "Total: 300.00 ms\nMean: 150.00 ms\np50: 100.00 ms\np95: 200.00 ms" in responses_text
    assert "Requests: 2\nPassed: 2\nFailed: 0" in decisions_text
    assert "Total: 50.00 ms\nMean: 25.00 ms\np50: 25.00 ms\np95: 25.00 ms" in decisions_text
    assert "Mean-latency speedup: 6.00x" in output.out
    assert "p50 speedup: 4.00x" in output.out and "p95 speedup: 8.00x" in output.out
    assert output.err == "" and "PRIVATE" not in output.out


def test_both_displays_fixed_decisions_model_when_responses_model_differs(capsys, cli_setup):
    assert live_router_eval.main([
        "--confirm-live", "--router-backend", "both", "--model", "responses-custom-model", "--case-id", SMOKE_CASE,
    ]) == 0
    assert [call.kwargs["model"] for call in cli_setup.factory.call_args_list] == ["responses-custom-model", "gpt-6-luna"]
    output = capsys.readouterr()
    assert "Responses model: responses-custom-model" in output.out
    assert "Decisions model: gpt-6-luna (fixed supported model)" in output.out
    assert "Models differ; this is not a same-model comparison." in output.out


@pytest.mark.parametrize("selector", [["--tag", "prompt_injection"], ["--all-cases"]])
def test_both_reuses_tag_and_all_case_selection(monkeypatch, capsys, selector):
    cases = load_router_cases()
    expected = [case for case in cases if "prompt_injection" in case.tags] if "--tag" in selector else cases
    received = []

    async def evaluate(selected, key, model, backend, samples):
        received.append((backend, selected))
        samples.extend([10.0] * len(selected))
        return await run_router_evaluation(selected, AsyncMock(side_effect=[case.expected_destination for case in selected]))

    monkeypatch.setattr(live_router_eval, "_evaluate", evaluate)
    assert live_router_eval.main(["--confirm-live", "--router-backend", "both", *selector]) == 0
    assert [backend for backend, _ in received] == ["responses", "decisions"]
    assert received[0][1] is received[1][1] and received[0][1] == expected
    assert "Router latency comparison" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ["accuracy", "missing_timing"])
def test_failed_or_incomplete_backend_cannot_produce_a_speedup_comparison(monkeypatch, capsys, failure):
    calls = []

    async def evaluate(cases, key, model, backend, samples):
        calls.append(backend)
        if failure != "missing_timing" or backend == "responses":
            samples.extend([10.0] * len(cases))
        destination = "verification" if failure == "accuracy" and backend == "responses" else "policy"
        return await run_router_evaluation(cases, AsyncMock(return_value=destination))

    monkeypatch.setattr(live_router_eval, "_evaluate", evaluate)
    assert live_router_eval.main(["--confirm-live", "--router-backend", "both", "--case-id", SMOKE_CASE]) == 1
    assert calls == ["responses", "decisions"]
    output = capsys.readouterr()
    assert output.out.count("Router evaluation") == 2
    assert "Router latency comparison" not in output.out
    assert "speedup:" not in output.out
    if failure == "missing_timing":
        assert output.err == "Router latency comparison incomplete; no comparison was produced.\n"


@pytest.mark.parametrize("case", ["missing_confirmation", "missing_key", "unknown_case", "unknown_tag", "empty_cases"])
def test_both_requires_credentials_confirmation_and_valid_selection_before_network(monkeypatch, capsys, cli_setup, case):
    args = ["--confirm-live", "--router-backend", "both", "--case-id", SMOKE_CASE]
    if case == "missing_confirmation":
        args.remove("--confirm-live")
    elif case == "missing_key":
        monkeypatch.delenv("OPENAI_API_KEY")
    elif case == "unknown_case":
        args[-1] = "unknown-case"
    elif case == "unknown_tag":
        args[-2:] = ["--tag", "unknown-tag"]
    else:
        monkeypatch.setattr(live_router_eval, "load_router_cases", list)
        args[-2:] = ["--all-cases"]

    with pytest.raises(SystemExit) as error:
        live_router_eval.main(args)
    assert error.value.code == 2
    live_router_eval.AsyncOpenAI.assert_not_called()
    cli_setup.factory.assert_not_called()
    output = capsys.readouterr()
    assert output.out == "" and "PRIVATE_API_KEY" not in output.err


def test_both_failure_never_prints_partial_comparison_or_upstream_content(monkeypatch, capsys):
    async def evaluate(cases, key, model, backend, samples):
        if backend == "decisions":
            raise RuntimeError("PRIVATE_API_KEY PRIVATE_PROMPT PRIVATE_UPSTREAM_BODY")
        samples.extend([10.0] * len(cases))
        return await run_router_evaluation(cases, AsyncMock(return_value="policy"))

    monkeypatch.setattr(live_router_eval, "_evaluate", evaluate)
    assert live_router_eval.main(["--confirm-live", "--router-backend", "both", "--case-id", SMOKE_CASE]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Live Router evaluation failed; no report was produced.\n"
