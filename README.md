# PawLine

PawLine is a minimal scaffold for an after-hours veterinary intake and escalation AI agent.

This repository contains the implementation foundation for the after-hours veterinary intake flow. The customer lookup, scheduling, routing, handoff, and clinic-policy RAG foundations are implemented. The broader agent, voice interface, database, and frontend remain future work.

## Local setup

This project uses `uv` as the package manager and runner.

```bash
cd pawline
uv sync --dev
uv run pytest -q
```

## Full-stack local run

Terminal 1: PawLine app

```bash
LEGACY_CRM_BASE_URL=http://127.0.0.1:8001 LEGACY_CRM_API_KEY=dev-crm-key SCHEDULER_BASE_URL=http://127.0.0.1:8002 SCHEDULER_JWT_SECRET=dev-scheduler-secret-at-least-32-bytes HANDOFF_BASE_URL=http://127.0.0.1:8003 HANDOFF_WEBHOOK_SECRET=dev-handoff-secret uv run uvicorn app.main:app --reload --port 8000
```

Terminal 2: mock CRM service

```bash
MOCK_CRM_API_KEY=dev-crm-key uv run uvicorn mock_services.legacy_crm_api:app --reload --port 8001
```

Terminal 3: mock scheduling service

```bash
MOCK_SCHEDULER_JWT_SECRET=dev-scheduler-secret-at-least-32-bytes uv run uvicorn mock_services.scheduling_api:app --reload --port 8002
```

Terminal 4: mock handoff service

```bash
MOCK_HANDOFF_WEBHOOK_SECRET=dev-handoff-secret uv run uvicorn mock_services.handoff_api:app --reload --port 8003
```

## Health check

Once the PawLine app is running:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status": "ok"}
```

## Runtime secrets (7G3)

The runtime reuses `app.infisical.get_openai_api_key()`. Configure only
`INFISICAL_CLIENT_ID`, `INFISICAL_CLIENT_SECRET`, and `INFISICAL_PROJECT_ID`
securely in the shell or deployment environment. These bootstrap credentials
authenticate the workload using Machine Identity/Universal Auth. Give that
identity only read access to the `OPENAI_API_KEY` secret in `dev` at `/` on
Infisical Cloud (`https://app.infisical.com`).

On the first runtime or policy-service initialization, Python runs the helper
in a worker thread and uses its returned key to construct the shared OpenAI
client. Later HTTP requests and caller turns reuse that client; shutdown closes
it. The helper also caches successful retrieval per process. Restart the app
after rotating the key. A missing configuration or failed lookup leaves the
agent service unavailable, with no fallback to an environment OpenAI key. Health
and direct non-OpenAI integration routes can still run without these credentials.
Secrets stay in memory as credentials and never enter prompts, conversation
state, workflow events, or API responses. Errors exclude SDK response details.
Live evaluation CLIs retain their existing environment-key behavior.

After setting the three bootstrap variables, test retrieval alone with
`uv run python -m app.infisical`, or start the real application:

```bash
uv run --env-file .env.example --env-file .env uvicorn app.main:app --port 8000
```

For this local demo, `.env.example` supplies the existing mock-service settings;
the later `.env` supplies your private Infisical configuration and overrides any
matching demo settings. Do not use demo credentials for a deployed service.
The Infisical bootstrap variables alone are insufficient for the full runtime:
`LEGACY_CRM_API_KEY`, `SCHEDULER_JWT_SECRET`, and `HANDOFF_WEBHOOK_SECRET` must
also be configured for their respective services. No environment OpenAI key is
required or used.

Then initialize the runtime:

```bash
curl -X POST http://127.0.0.1:8000/conversations
```

Initialization loads and indexes the policy documents using the shared client's
embeddings API, so it requires working OpenAI embeddings access as well as
Infisical access. A smoke-test success verifies retrieval, not OpenAI API access
or the other integrations' configuration. Initialization failures keep the
sanitized HTTP 503 and close the unused client; later requests can retry.
The app does not automatically load `.env`. The standalone smoke check prints
only success or a sanitized error and makes no OpenAI request. Automated tests
mock the helper and embedding requests.

## Customer lookup check

```bash
curl "http://127.0.0.1:8000/customers/lookup?phone=3212222222"
```

## Appointment slot lookup

```bash
curl -H "Authorization: Bearer <ignored-for-local-demo>" "http://127.0.0.1:8000/appointments/slots?pet_id=pet_2001&appointment_type=same_day"
```

## Appointment booking

```bash
curl -X POST http://127.0.0.1:8000/appointments/bookings \
  -H "Content-Type: application/json" \
  -d '{
    "slot_id": "slot_same_day_01",
    "pet_id": "pet_2001",
    "confirmed_by_caller": true
  }'
```

## Handoff webhook note

Direct manual curl testing for the handoff webhook is inconvenient because the signature must match the exact request-body bytes. Automated tests are sufficient for this checkpoint.

## PawLine handoff example

The PawLine app computes the routing decision internally before it sends a handoff webhook. The caller sends intake information only; it does not send a routing_decision, case_id, status, or HMAC signature.

```bash
curl -X POST http://127.0.0.1:8000/handoffs \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv-demo-001",
    "verified_customer_id": "customer_1001",
    "selected_pet_id": "pet_2001",
    "original_concern": "labored breathing",
    "intake_answers": {
      "difficulty_breathing": true,
      "uncontrolled_bleeding": false,
      "collapsed_or_unresponsive": false,
      "known_toxin_exposure": false,
      "rapidly_worsening": false
    }
  }'
```

## Part 5: clinic policy RAG foundation

PawLine now has a narrow, auditable policy-answer flow for administrative clinic questions.

Policy Markdown
-> chunks
-> embeddings
-> in-memory index
-> query embedding
-> cosine-similarity retrieval
-> relevance gate
-> grounded generation
-> verified citations

Key behaviors:

- The model never receives the entire policy corpus.
- The model receives only the top retrieved chunks that are most similar to the user question.
- Similarity thresholding is an initial safeguard that must be evaluated in future Part 7 tuning.
- PawLine constructs citations from validated chunk IDs rather than trusting the model to invent source names.
- Medical safety routing remains deterministic and separate from this policy-answer feature.
- This is a fictional administrative policy layer for clinic operations, not veterinary diagnosis or treatment advice.

### Environment variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
export OPENAI_POLICY_MODEL="gpt-6-luna"
export POLICY_MIN_SIMILARITY="0.45"
export POLICY_TOP_K="3"
```

Automated tests never call OpenAI and do not require an API key.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/policies/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can I cancel my appointment?"
  }'
```

Expected response shape:

```json
{
  "question": "Can I cancel my appointment?",
  "answer": "The clinic policy allows appointment cancellations with notice.",
  "status": "answered",
  "citations": [
    {
      "chunk_id": "appointments::cancellation-policy",
      "document_id": "appointments",
      "document_title": "Appointments and Cancellations",
      "section_title": "Cancellation Policy",
      "source_name": "appointments_and_cancellations.md"
    }
  ]
}
```

If the evidence is weak or missing, the route returns `status: "insufficient_context"` with a safe fallback answer and no citations instead of inventing a policy answer.

## Prompt-driven conversations

The conversation endpoints use `PromptAgentRuntime` to send each caller message
through the Router and one specialist. The specialist can request approved tools;
Python executes their prepared handlers and saves the conversation state.

The runtime and conversation store are shared across HTTP requests in this process.
The Router and runtime-configured specialists share one OpenAI client. The prepared
Policy Agent keeps its existing retrieval dependencies. Tests use fake model and
service clients without making external API calls.

## Part 6A: conversation memory foundation

PawLine now includes a minimal internal conversation-memory model for future multi-turn agent work.

- `ConversationState` is one snapshot of what PawLine knows about a single call or conversation.
- `InMemoryConversationStore` keeps snapshots by conversation ID in the current process only.
- This store is intentionally process-local for the portfolio: it is not a database and it disappears when PawLine restarts.
- The future orchestration layer will read a state, perform an action, update it, and save it again.
- This checkpoint contains no LLM call, no agent decision-making, and no message interpretation.

The state holds the fields that allow a future orchestrator to continue a conversation across multiple caller messages, including the verified customer, selected pet, intake answers, routing decision, appointment selection, booking data, and handoff receipt.

### Implementation notes

- Policy documents are loaded from the fictional clinic policy Markdown corpus in `knowledge/clinic_policies`.
- `PolicyRetriever` ranks policy chunks by cosine similarity and keeps the in-memory index stable for repeated requests.
- The answer-generation boundary is isolated behind a provider protocol so tests can use fake embeddings and fake policy answers without real network calls.
- PawLine validates the model's supporting chunk IDs before constructing any public citation objects.

## Notes

- Python 3.12+ is targeted for the project.
- Dependencies are managed with `uv` instead of `pip`.
- The mock CRM is intentionally separate from the PawLine app and is only used for local development.
