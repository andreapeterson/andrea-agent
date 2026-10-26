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
- **Argument accuracy:** exact JSON arguments, across expected-tool cases only.
  Object key order is ignored; types, missing keys, nulls, and exact strings
  (including explanation/query strings) matter. Arguments are graded independently
  of tool-name correctness, so one mistake need not fail both checks.
- **Complete-case accuracy:** every applicable check must pass. Speech cases
  require no tool or arguments; spoken wording is never compared.

Empty metric denominators receive 0. Errors in the injected callable propagate.
7D2 tests the evaluation machinery using fakes; those scores do not measure
the real model or prove execution safety. Expected labels and human reasons
stay out of model input.

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

The evaluation deliberately stops before executing a tool. It creates no handlers
or integration clients, calls no CRM, scheduler, policy retrieval, or handoff,
sends no `function_call_output`, and makes no second model call. It evaluates
tool selection and arguments, not the complete tool loop or final post-tool reply.
It does not establish real model grounding or production execution safety.

Malformed responses and request failures stop the run with a generic safe error.
The command exits with status 1 for a failed case or an incomplete live run; it
prints no API key, full prompt, or raw response, and closes the client afterward.
Live scores can vary when the model or prompt changes. Normal pytest uses fake
OpenAI clients only and never opts into paid evaluation.
