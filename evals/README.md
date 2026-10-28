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
enum values (including `request_handoff.reason`), all intake fields, and scheduling
and booking arguments remain exact. Object key order is ignored. No business
values are normalized or approved by the semantic judge.

Only `search_policy.query` and `request_handoff.explanation` can receive semantic
grading. Pass an async `semantic_grader(case, field, expected_text, actual_text)`
returning `SemanticGrade(equivalent, reason)` to `run_specialist_evaluation`.
Without an injected grader, the offline scorer retains exact JSON comparison.
It never constructs an OpenAI client; offline tests inject fake graders.

The grader is called only when an approved nonblank string differs exactly,
the requested tool matches, and all keys, structure, and other fields already
match exactly. Exact prose matches need no judge. An argument score passes only
if every exact check passes and the prose preserves the same operational meaning
without adding facts unsupported by the caller message or recorded context.
For example, paraphrasing clinic hours can pass; switching to prices cannot.
A changed handoff reason fails without asking the judge to approve the explanation.
Semantic failures include the field name and the judge's short reason.
Judge errors propagate and abort evaluation rather than counting as model failures.

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
message, and allowlisted context as untrusted user data, separate from its grading
instructions. It uses a small `SemanticGrade` schema through
[Responses structured parsing](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).
It checks operational meaning and unsupported added facts, without executing tools.

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
