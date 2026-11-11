# Router evaluation

`router_cases.jsonl` contains 25 fictional caller cases, five per destination.
The expected routes and reasons are human-authored expectations. Context uses
the existing `PromptContext`; recent messages are earlier saved history only.

Unit tests verify deterministic Python behavior. Evaluation cases measure model
behavior against human expectations. In 7D1, tests inject fake async callables
to verify loading, grading, and reporting. Their scores do **not** measure the
real Router's accuracy. No OpenAI or external-service calls occur in these tests.

The grader compares destination strings exactly because the Router has five
fixed destinations. It does not compare model wording or grade the Router's
free-text reason. Reports include overall and per-destination accuracy as
fractions from 0 to 1; empty groups receive 0. No results are stored automatically.
Callable errors propagate rather than being reported as successful evaluations.

An expected `human_handoff` destination selects the conversational specialist;
it does not authorize a queued request or transfer. The specialist may request
a tool, and the Python handler decides whether to execute a real handoff.

Use the runner from Python with an injected `async route_case(context,
user_message) -> str`, then print `format_router_report(report)`. For a local
command, provide that callable explicitly:

```bash
uv run python -m evals.router_eval --route-case my_local_router:route_case
```

The module must be importable from the project root. An optional `--cases PATH`
selects another JSONL file. There is no default routing implementation or client.
The loader skips whitespace-only lines and rejects malformed rows, duplicate IDs,
blank metadata, invalid context, and unsupported expected destinations.

## Live Router evaluation (7D3A)

This is opt-in because it requires credentials, network access, time, and API cost.
Set `OPENAI_API_KEY` in your environment; do not put its value in source files.
The command requires `--confirm-live` plus explicit case IDs or `--all-cases`.
It never selects the full dataset automatically.

Single smoke case:

```bash
uv run python -m evals.live_router_eval --confirm-live --case-id policy_clinic_hours_unverified
```

Several selected cases:

```bash
uv run python -m evals.live_router_eval --confirm-live --case-id policy_clinic_hours_unverified --case-id verification_injection_pretend_verified
```

Full Router dataset:

```bash
uv run python -m evals.live_router_eval --confirm-live --all-cases
```

`--model` takes precedence over `OPENAI_ROUTER_MODEL`; the fallback is PawLine's
current Router default, `gpt-6-luna`. Unknown case IDs are rejected before creating
a client. Repeated IDs are evaluated once, in dataset order.

Each selected case makes one OpenAI Router request, sequentially. The JSONL
supplies the expected destination; OpenAI returns the actual destination through
the existing `OpenAIRouter.route()`. The existing deterministic scorer compares
them and prints its existing report. The adapter renders the real Router prompt
from context and passes the current caller message separately as input.

The process exits with status 1 if any evaluated case fails or the live run cannot
complete. It prints no API key, full prompt, or raw model response. It constructs
no specialists and executes no handlers or integration calls beyond the Router.
The async client is closed after the run, following the
[OpenAI SDK client lifecycle guidance](https://developers.openai.com/api/reference/python#managing-http-resources).

Live results can vary when the model or prompt changes. Normal pytest tests use
fake OpenAI clients only; running pytest does not opt into a paid evaluation.

## Specialist first actions (7D2)

Router eval asks, “Which specialist?” Specialist eval asks, “Should this
specialist speak or request a tool, and with what arguments?”
`specialist_cases.jsonl` contains 15 fictional cases, three per specialist.
The loader reuses `PromptContext` and the real Pydantic tool-argument models,
including required null fields. It also checks each specialist's approved tools.

Pass an async `run_case(case)` to `run_specialist_evaluation`; return a
`SpecialistFirstAction(action, tool_name, arguments)`. The runner records one
first action per case. It never constructs clients, runs agents, retrieves
policy, or executes handlers. `format_specialist_report` prints the report.

Scores are separate, overall and per specialist:

- **Action accuracy:** tool versus speech, across all cases.
- **Tool-name accuracy:** exact tool name, across expected-tool cases only.
- **Argument accuracy:** exact JSON arguments with the two optional semantic checks
  described below, across expected-tool cases only. Arguments are graded independently
  of tool-name correctness, so one mistake need not fail both checks.
- **Complete-case accuracy:** every applicable check must pass. Speech cases
  require no tool or arguments; spoken wording is never compared.

Empty metric denominators retain numeric accuracy 0 in the report dictionary.
Terminal scores display `N/A (0 applicable cases)` when the selected cases did
not require that check. Errors in the injected callable propagate.
7D2 tests the evaluation machinery using fakes; those scores do not measure
the real model or prove execution safety. Expected labels and human reasons
stay out of the specialist's model input.

### Exact and semantic argument grading

Tool names, argument keys and structure, JSON types, phone numbers, IDs, booleans,
enum values (including `request_handoff.reason`), intake answers, and scheduling
and booking arguments remain exact. Object key order is ignored.

Only `record_intake.original_concern` receives limited deterministic normalization:
trim surrounding whitespace, collapse repeated internal whitespace, and remove
terminal sentence marks (`.`, `!`, `?`). The remaining text is compared exactly.
Words, case, and internal punctuation still matter; paraphrases and inferred
diagnoses fail. This field never goes to the semantic judge.

Only `search_policy.query` and `request_handoff.explanation` can receive semantic
grading. Pass an async `semantic_grader(case, field, expected_text, actual_text)`
returning `SemanticGrade(equivalent, reason)` to `run_specialist_evaluation`.
Without an injected grader, the offline scorer retains exact JSON comparison
apart from the limited concern normalization above.
It never constructs an OpenAI client; offline tests inject fake graders.

The grader is called only when an approved nonblank string differs exactly,
the requested tool matches, and all keys, structure, and other fields already
match exactly. Exact prose matches need no judge. An argument score passes only
if every exact check passes and the prose preserves the same operational meaning
without adding facts unsupported by the caller message or recorded context.
For example, paraphrasing clinic hours can pass; switching to prices cannot.
A changed handoff reason fails without asking the judge to approve the explanation.
For handoff explanations, the reference specifies required operational meaning,
not a maximum permitted wording. Additional language may be accepted when directly
supported by the caller message, trusted context, or exact structured arguments.
Recorded urgent intake with `reason=urgent_intake` supports saying urgent human
assistance is required. The reason alone does not establish clinical urgency.
Diagnoses and treatment claims are rejected. Invented symptoms, identities,
outcomes, and other unsupported additions still fail.
Semantic failures include the field name and the judge's short reason.
Judge errors propagate and abort evaluation rather than counting as model failures.

Semantic judge context is field-specific. Policy-query grading receives an empty
context: only the current caller question and expected/actual queries are used,
with no conversation identity, pet, concern, intake, or routing data. Handoff
explanation grading receives only present trusted facts: `customer_first_name`,
`known_pets`, `selected_pet`, `original_concern`, non-null intake answers, and
`routing_level`. Null/default values and unrelated context are omitted. Unknown
semantic tool/field combinations raise an evaluation configuration error.

Semantic grading is evaluation-only. It changes no production agents, prompts,
tools, handlers, or expected dataset wording.

## Live specialist first actions (7D3B)

This is opt-in because it requires credentials, network access, time, and API cost.
Set `OPENAI_API_KEY` in your environment. The command requires `--confirm-live`
plus explicit case IDs or `--all-cases`; it never selects all cases automatically.

Speech smoke case:

```bash
uv run python -m evals.live_specialist_eval --confirm-live --case-id verification_missing_phone
```

Tool smoke case:

```bash
uv run python -m evals.live_specialist_eval --confirm-live --case-id verification_lookup_supplied_phone
```

All specialist cases:

```bash
uv run python -m evals.live_specialist_eval --confirm-live --all-cases
```

`--model` takes precedence over `OPENAI_SPECIALIST_MODEL`; the fallback is PawLine's
current specialist default, `gpt-6-luna`. Unknown IDs are rejected before creating
a client. Repeated IDs run once, sequentially in dataset order.

Each selected case makes one paid OpenAI Responses request using that specialist's
real rendered prompt and approved production tool definitions. The current caller
message is a separate user input item. The adapter observes speech or one function
call, decodes its arguments, and returns `SpecialistFirstAction` to the existing
scorer. The JSONL supplies the expected action, tool, and arguments; the existing
report compares them with the model's first action.

The live evaluator injects an OpenAI-backed semantic judge. Each differing approved
prose field that passes the exact checks adds one paid judge request, with extra
latency and API cost (currently at most one judge request per selected case).
`OPENAI_EVAL_JUDGE_MODEL` selects the judge model; otherwise it uses the configured
live specialist model. The judge receives the expected and actual text, caller
message, field-specific context described above, and (for handoff explanations)
the exact handoff reason as user data, separate from its grading
instructions. It uses a small `SemanticGrade` schema through
[Responses structured parsing](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).
It checks operational meaning and unsupported added facts, without executing tools.
Conversation history, appointment data, tool lists, and internal routing details
are omitted from the judge's context.

The evaluation deliberately stops before executing a tool. It creates no handlers
or integration clients, calls no CRM, scheduler, policy retrieval, or handoff,
sends no `function_call_output`, and makes no specialist continuation call. It evaluates
tool selection and arguments, not the complete tool loop or final post-tool reply.
It does not establish real model grounding or production execution safety.

Malformed responses and request failures stop the run with a generic safe error.
The command exits with status 1 for a failed case or an incomplete live run; it
prints no API key, full prompt, or raw response, and closes the client afterward.
Live scores can vary when the model or prompt changes. Normal pytest uses fake
OpenAI clients only and never opts into paid evaluation.

## Final specialist responses (7D5, offline harness)

`specialist_eval.py` grades the first model action: speech or a tool request and
its arguments. The separate `response_eval.py` grades final caller-facing language
after a controlled tool result. It does not grade tool selection or execute a tool.

`response_cases.jsonl` starts with five fictional cases, one per specialist:
customer not found, grounded clinic hours, incomplete intake, available slots
without a booking, and a queued handoff without a live connection. Each
`ResponseEvalCase` contains `case_id`, `specialist`, `user_message`, `context`
using `PromptContext`, nullable `tool_name`, `tool_arguments`, and `tool_result`,
human-written `criteria` with stable IDs and descriptions, and `tags`. All three
tool fields must be supplied together or all be null. Arguments are validated
with the named tool's existing Pydantic argument model and are not rewritten.
Tool results are controlled JSON fixtures;
they are never fetched from CRM, scheduling, policy retrieval, or handoff services.
The JSONL loader skips blank lines and rejects invalid rows or duplicate normalized
case IDs with filename and line number.

Supply two async callables explicitly:

- `generate_response(case) -> str`: produces nonblank final caller-facing text.
- `grade_response(case, response) -> ResponseJudgeResult`: returns `passed`, a
  nonblank `reason`, and `criteria_results`, each with a `criterion_id` and boolean
  `passed`. Every case criterion must appear exactly once; order does not matter.

For example, inside your own async evaluation code:

```python
from evals.response_eval import (
    format_response_report,
    load_response_cases,
    run_response_evaluation,
)

report = await run_response_evaluation(
    load_response_cases(), generate_response, grade_response
)
print(format_response_report(report))
```

A complete case passes only when the judge's overall decision and every human
criterion pass. A failed criterion cannot be overridden by `passed=True`.
Blank responses, incomplete or duplicate criterion judgments, and generator or
judge errors abort evaluation instead of counting as model failures.

Reports contain total, passed, failed, and `pass_rate`, summaries by specialist,
per-case responses and criterion results, and failures. The terminal formatter
shows failed case IDs, judge reasons, and failed criterion IDs and descriptions.
Empty groups retain numeric pass rate 0 in the dictionary and display `N/A`.
Reports are returned in memory and are not saved automatically.

This offline harness constructs no OpenAI clients, production agents, or handlers.
Tests inject fake response generators and predetermined fake judgments; they
verify loading, scoring, and reporting, **not a real model's final-response quality
or grounding**. The opt-in live adapter below supplies real model callables;
normal pytest makes no paid evaluation requests.

## Live final-response evaluation (7D5B)

`live_specialist_eval` evaluates the real model's first action, tool selection,
and arguments. `response_eval` scores final caller-facing speech.
`live_response_eval` assumes the expected tool call already occurred and connects
that harness to real specialist and judge models using controlled post-tool input.

Set `OPENAI_API_KEY` in the environment. Every live command requires
`--confirm-live` and exactly one selection mode: repeatable `--case-id` or
`--all-cases`. Unknown IDs are rejected before client creation. Repeated IDs run
once, in dataset order, sequentially. The entire dataset is never selected by default.

Single smoke case:

```bash
uv run python -m evals.live_response_eval --confirm-live --case-id verification_customer_not_found
```

Several cases:

```bash
uv run python -m evals.live_response_eval --confirm-live --case-id policy_grounded_clinic_hours --case-id handoff_queued_not_connected
```

All five response cases:

```bash
uv run python -m evals.live_response_eval --confirm-live --all-cases
```

`--model` overrides `OPENAI_SPECIALIST_MODEL`, falling back to PawLine's current
specialist default, `gpt-6-luna`. `OPENAI_EVAL_JUDGE_MODEL` selects the judge;
otherwise it uses the selected specialist model. One OpenAI client is shared
and closed after the run.

Generation renders the real specialist prompt and sends its real tool definitions
with one specialist continuation request. Input contains the caller message, a
synthetic preserved `function_call` containing the fixture's `tool_name` and
serialized `tool_arguments`, and its matching `function_call_output` containing
the serialized `tool_result`. Both items share a deterministic case-scoped `call_id`.
No tool selection is requested again, and neither `run_agent_turn` nor any handler
executes. The response must be nonblank final speech; any new tool request fails
safely without execution or another specialist request. A speech-only case sends
just the caller message and follows the same final-speech requirement.
First-action evaluation tests model argument choices; this adapter validates
predefined arguments only to ensure the injected completed call is well-formed.

No production handler or integration is constructed or called. There is no CRM,
scheduler, retriever, or handoff access, and no `ConversationState` is created or
modified. Real production prompts and agent behavior are unchanged.

Specialist and judge contexts use explicit allowlists. Policy receives no identity
or intake context; the other specialists receive only relevant caller/pet facts,
intake or routing facts, and appointment context needed by that specific scheduling
operation. Conversation history, credentials, IDs from private state, idempotency
keys, receipts, and unrelated appointment/intake data are omitted. Fixture arguments
and results with forbidden private fields are rejected before a live request,
rather than silently changing their values. A pet ID is permitted only
when a human criterion explicitly has ID `exact_pet_id`. The initial scheduling
fixture omits its unused pet ID; its grading criteria are unchanged.

The judge receives only final text, caller message, specialist, all original criterion
IDs and descriptions, controlled tool name, arguments and result when present, and
the projected context.
It uses [Responses structured parsing](https://developers.openai.com/api/docs/guides/structured-outputs)
into `ResponseJudgeResult`, grades every
criterion once, allows equivalent wording, and treats all supplied text as data
rather than instructions. The existing scorer enforces complete criterion coverage
and requires every criterion to pass.

Live runs require credentials, network access, time, and API cost. Each current
tool-result case makes one specialist continuation call and one judge call. A future
speech-only case makes one normal specialist request and one judge call. Errors can
stop a run earlier. The conversation evaluation below tests the complete first model
call → handler → second model call sequence; this controlled post-tool evaluation
does not test real integrations or complete production conversations.

The command prints the existing response report and exits with status 1 for any
failed case or aborted evaluation. Error output excludes credentials, full prompts,
raw OpenAI responses, and upstream exception details. All unit tests use fake OpenAI
clients and do not opt into live calls.

## Conversation workflows (7D6A, offline harness)

The component evaluations test one layer: `router_eval` checks specialist selection,
`specialist_eval` checks first actions and arguments, and `response_eval` checks
final caller-facing wording. `conversation_eval` checks the expected multi-turn
structure instead, without duplicating response-language grading.

`conversation_cases.jsonl` contains three fictional golden workflows: an unverified
clinic-hours question, verification followed by urgent intake and handoff, and
verification followed by routine intake, slot search, selection, and explicit booking
confirmation. They use demo identifiers. The policy case retains the runtime's
initial `verifying_customer` phase without requiring or performing verification.

Inject an async `run_conversation(case) -> ConversationRunObservation` into
`run_conversation_evaluation(load_conversation_cases(), run_conversation)`, then
print `format_conversation_report(report)`. Observations contain specialist and
ordered tool lists for each caller turn, plus the final phase and verification,
booking, and handoff outcomes. Both models validate specialist/tool ownership,
turn counts, and consistent completion outcomes. Loader errors include filename
and line number; duplicate normalized case IDs are rejected.

The scorer compares specialist sequence, tool sequence (including call order within
each turn), final phase, verification, booking, and handoff separately. A complete
case passes only if all six checks pass. Every check applies to every case:
expected `false` outcomes also enforce the absence of unintended actions. Reports
include counts, pass rate, per-check accuracy, per-case results, and exact
expected-versus-actual failure details. Empty metrics display `N/A`; their numeric
accuracy remains zero. Invalid observations and runner errors propagate.

7D6A uses injected fake observations. It does **not** test a real model, execute
production tools, construct the runtime or a conversation store, or make external
requests. The controlled live adapter below adds real model calls in 7D6B. Response wording remains
covered by `response_eval`.

## Live conversation workflows (7D6B)

`conversation_eval.py` is the model-independent scorer. `live_conversation_eval.py`
runs the real `PromptAgentRuntime` directly, without FastAPI: Router → specialist →
production tool handler → controlled integration → tool result → specialist
continuation → state save. Unlike `live_response_eval`, it intentionally runs the
complete Responses API tool loop and preserves state across caller turns.

Set `OPENAI_API_KEY`, `OPENAI_ROUTER_MODEL`, and `OPENAI_SPECIALIST_MODEL` in your
environment. Both model names must be explicitly configured; `--router-model`
and `--specialist-model` override the corresponding environment variables. The
command requires `--confirm-live` and exactly one selection mode, repeatable
`--case-id` or `--all-cases`. Unknown IDs fail before client creation; repeated
IDs run once in dataset order. It never selects all cases automatically.

Single policy workflow:

```bash
uv run python -m evals.live_conversation_eval --confirm-live --case-id unverified_policy_hours
```

Several workflows:

```bash
uv run python -m evals.live_conversation_eval --confirm-live --case-id verified_urgent_intake_handoff --case-id verified_routine_booking
```

All three workflows:

```bash
uv run python -m evals.live_conversation_eval --confirm-live --all-cases
```

Each case gets a unique demo conversation ID, a fresh in-memory store, and fresh
evaluation-only integrations. CRM recognizes only the two fictional dataset phone
numbers and returns Avery Demo with the dog Maize. Policy search returns controlled
demo clinic-hours evidence without embeddings. Scheduling supplies demo routine
or same-day slots and deterministic booking confirmations; repeated identical
booking requests with the same idempotency key reuse the confirmation. Handoff
returns a fictional queued receipt. No production HTTP integration client is
constructed, and no real CRM, policy, scheduler, or handoff service is contacted.
The only network client is OpenAI, using the official API base URL.

Only caller turns enter the runtime. Expected sequences, outcomes, case IDs, and
tags stay out of model inputs. The optional structured trace hook records turn,
model-request, specialist, and tool lifecycle events. The evaluator uses only
`specialist_selected` and `tool_requested` to extract its sequences, so completion
events never count as extra tool calls. Trace metadata includes conversation ID,
turn number, event type, and known specialist/tool names, prompt version, model,
elapsed duration, execution status, and validated optional token counts on completed
model requests. The hook is disabled by default and scoped
to the current task and turn; observer failures cannot break the workflow. It records no caller text,
prompts, tool arguments/results, identity facts, credentials, or raw responses.
Final phase, verification, booking, and handoff outcomes come from saved
`ConversationState`, never from model prose.

Wrong routes or approved tool choices are scored as observed; the adapter does not
rewrite them to match expectations. Runtime/model/integration errors abort with a
generic safe message. If early completion causes a later caller turn to skip the
Router, the run aborts rather than inventing a specialist for that turn. Reports
show only the existing structural scores and mismatches. The command exits with
status 1 for a failed case or an aborted run and does not print prompts, tool data,
raw model responses, credentials, or exception chains.

Live runs may make many paid OpenAI calls: ordinarily one Router call per turn,
one specialist call, and one continuation per tool execution. Tool loops can add
calls up to their existing production limit. Trace events include operation durations
and available token counts. Pure turn summaries and optional estimates using injected
rates are documented below; they do not alter evaluation scoring or runtime behavior.
There is no LLM judge or wording grade. Final response wording remains evaluated separately
by `response_eval.py`. Unit tests use fake OpenAI clients, run the real workflow
with controlled integrations, and make no paid requests.

See [safe agent traces](../docs/agent_tracing.md) for event lifecycles and emission points.

## Failure localization (7E3A)

The evaluation harnesses measure behavior. `failure_attribution.py` adds a pure,
deterministic analysis layer that localizes failures to the observed component;
it does not diagnose a guaranteed root cause or change any score.

```python
from evals.failure_attribution import (
    attribute_report_failures,
    format_failure_attribution_report,
    summarize_failure_attributions,
)

attributions = attribute_report_failures(report, eval_type="specialist")
summary = summarize_failure_attributions(attributions)
print(format_failure_attribution_report(attributions))
```

Choose `router`, `specialist`, `response`, or `conversation` explicitly. The layer
validates report counts and required result fields without changing the report.
It returns a tuple of frozen models ordered by case ID and fixed check/criterion
order. Passing cases produce none; one case can produce several observations.

Response attributions use only safe human-written criterion IDs/descriptions.
An overall failed response grade without failed criteria receives a generic
response attribution. Judge explanations, generated text, prompts, caller data,
tool payloads, credentials, and exception details are excluded. The formatter
shows only component counts and case IDs; counts represent observations rather
than unique failed cases.

Localization is evidence for investigation, not proof of root cause. Developers
still inspect the suggested prompt, schema, fixture, handler, or state transition.
This layer makes no model requests and performs no automatic prompt rewriting.

The live Router, specialist, response, and conversation commands now append a
localization section only when the evaluation has failures. The original detailed
report appears first, followed by a summary in this format:

```text
Failure localization
Total attributions: 2
tool_arguments: 1 (cases: example_case)
response_generation: 1 (cases: another_case)
```

Localization makes no additional model call and requires no new flag. Passing
runs print only the original report. If localization rejects an inconsistent
report, the command returns 1 with a local safe error and prints no report data.
