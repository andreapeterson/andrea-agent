# Safe agent traces (7E1)

A trace describes execution without recording conversation content. `WorkflowEvent`
is a frozen dataclass with a typed event enum and a fixed set of metadata fields:
conversation ID, turn number, event type, and optional specialist, tool name,
prompt version, duration in milliseconds, model name, execution status, and three
optional token counters on completed model requests.
There is no free-form payload or exception field.

For one caller turn requiring one tool, events normally arrive in this order:

```text
turn_started
model_request_started       # Router
model_request_completed
specialist_selected
model_request_started       # Specialist's first request
model_request_completed
tool_requested
tool_completed
model_request_started       # Specialist receives the tool result
model_request_completed
turn_completed
```

Additional tools repeat the specialist request and tool events. Speech-only turns
skip the tool and continuation events. An exception ends the turn with
`turn_failed` instead of `turn_completed`. A model or tool operation that raises
still emits its corresponding completed event, with `status="failed"`.

| Emission point | Events |
| --- | --- |
| `PromptAgentRuntime.handle_turn`, under its lock after loading state | `turn_started`; `turn_completed` after saving the reply and state, or `turn_failed` when the scoped workflow raises |
| `OpenAIRouter.route`, around its SDK parse call | `model_request_started`, `model_request_completed` |
| `route_and_respond`, after the validated Router decision | `specialist_selected` |
| `run_agent_turn`, around each SDK create call | `model_request_started`, `model_request_completed` |
| `run_agent_turn`, around an accepted handler call and JSON serialization | `tool_requested`, `tool_completed` |

The initial greeting is not a caller turn. Input validation and loading the
conversation happen before the trace scope, since its turn number comes from
loaded state. An already-complete conversation emits only turn boundaries when
returning its existing reply; it does not start another model call or change state.

Durations use `perf_counter`, a monotonic clock. Turn duration excludes waiting for
the conversation lock and loading state. Model duration covers the SDK invocation,
including any SDK retries. Tool duration includes execution and JSON serialization.
`succeeded` means that operation returned without raising; it does not assert a
business outcome. For example, a tool can execute successfully and return
`verification_required` or no matching evidence. Response validation after an SDK
call may still fail the turn. Booking and handoff outcomes remain in trusted state.

The runtime's existing `workflow_observer` keyword accepts a synchronous callback
receiving each event. It defaults to `None`: no events are delivered or timed, and
there is no default sink. A callback such as `events.append` is enough to inspect a
trace locally. Keep callbacks fast; no vendor, database, or exporter is installed.

`ContextVar` binds the observer, conversation ID, turn number, and selected
specialist to the current async task. Nested scopes restore the previous binding;
`finally` resets it after success, failure, or cancellation. Disabling observation
also suppresses any surrounding scope for that runtime turn.

`_safe_emit` invokes the callback and swallows observer exceptions without logging
their details. Callback return values are ignored, and events are immutable. The
observer receives no state, clients, prompts, tool arguments/results, caller text,
identity facts, credentials, raw responses, or exceptions. Events are never added
to model input. This is secondary observation of decisions and execution; Router
decisions, Python handler guards, and state saves remain authoritative.

The live conversation evaluator filters `specialist_selected` and `tool_requested`
events to obtain ordered sequences. It continues to read final outcomes from
`ConversationState`; it does not inspect trace status or model prose for outcomes.

## Token usage and turn summaries (7E2)

Both Router and specialist request scopes call the same `extract_token_usage`
helper after the SDK returns. It reads only `response.usage.input_tokens`,
`output_tokens`, and `total_tokens`, supporting SDK objects and dictionary fakes.
Only nonnegative integers are copied; booleans, strings, floats, negative values,
missing fields, and properties that raise become `None`. Event construction also
rejects invalid token values and tokens attached to another event type. No raw
response is kept by the recorder, and no nested usage details are inspected.

The three counters accompany `model_request_completed`, including a zero count
when zero was explicitly reported. Failed requests without a returned response
have unknown usage. No observer means usage is not inspected. There is no extra
API call, tokenizer, or local estimate of missing tokens. Reported output tokens
can include non-visible generated tokens; they are not a count of visible words.
See the [official OpenAI token-counting documentation](https://developers.openai.com/api/docs/guides/token-counting).

`summarize_turn_trace(events)` is pure Python. Give it events from exactly one
finished turn. It checks turn boundaries, one conversation and turn number,
matching serial request/tool pairs, and finite nonnegative measurements. Invalid
or incomplete traces raise a validation error. It returns a frozen
`TurnTraceSummary` with:

- the selected specialist, unique model names and prompt versions in first-use order;
- request count and tool names in execution order, retaining repeated tools;
- sums of completed model durations and tool durations, without adding turn
  duration again;
- sums of each reported token counter and the completed/failed turn outcome.

If any request lacks a token counter, that aggregate counter is `None` and prints
as `unknown`. Other fully reported counters can still be summed. No requests means
known zero usage. Missing operation durations are also unknown rather than a
partial total. A failed turn can still contain usage from completed requests.
The request count represents SDK invocations, not individual SDK retry attempts.

One caller turn can include a Router request, the specialist's first request, and
one continuation after each tool result. All of these belong in its usage totals.
Neither the summary nor its formatter calls OpenAI or changes conversation state.
The runtime does not invoke them automatically; errors in a summary callback are
isolated by the same safe observer boundary.

For local inspection after collecting a complete turn:

```python
from app.agents.turn_trace_summary import (
    estimate_turn_cost,
    format_turn_trace_summary,
    summarize_turn_trace,
)

summary = summarize_turn_trace(events)
print(format_turn_trace_summary(summary))
# Optional, using a separately supplied pricing mapping:
estimate = estimate_turn_cost(events, pricing)
print(format_turn_trace_summary(summary, estimated_cost=estimate))
```

`pricing` maps model names to `ModelTokenPricing(input_per_million,
output_per_million)`. Both rates must be finite, nonnegative `Decimal` values in
**USD per 1,000,000 tokens**. The estimator sums each request's
`(input_tokens * input_rate + output_tokens * output_rate) / Decimal(1_000_000)`.
It returns a `Decimal` USD estimate, or `None` if any model lacks pricing or any
request has incomplete usage. It never substitutes one model's rate for another.
A turn with no model requests has a zero estimate.

No prices are embedded in application logic or fetched automatically. This basic
two-rate calculation does not account for caching tiers, discounts, hidden retry
charges, or other billing adjustments. It is labeled **Estimated cost (USD)**,
not a billed amount. Pricing is optional; the runtime and summary work without it.

The formatter reads only named summary fields. It never receives caller messages,
rendered prompts, tool arguments or results, customer/pet IDs, credentials, raw
responses, or exception details. Cached input details and reasoning details are
also excluded from events and summaries.
