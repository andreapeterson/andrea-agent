# Safe agent traces (7E1)

A trace describes execution without recording conversation content. `WorkflowEvent`
is a frozen dataclass with a typed event enum and a fixed set of metadata fields:
conversation ID, turn number, event type, and optional specialist, tool name,
prompt version, duration in milliseconds, model name, and execution status.
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
