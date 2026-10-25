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

A later opt-in live evaluation will supply an adapter that renders the existing
Router instructions, calls the real Router with the caller message as input,
and returns the decision's destination. It will reuse this dataset and grader;
7D1 does not implement or invoke that adapter.

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
7D2 tests the evaluation machinery using fakes; its scores do not yet measure
the real model or prove execution safety. A later opt-in live adapter will run
these same cases against OpenAI and capture the first action without executing
the requested tool. Expected labels and human reasons must stay out of model input.
