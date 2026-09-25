# PawLine

PawLine is a minimal scaffold for an after-hours veterinary intake and escalation AI agent.

This repository contains the initial implementation foundation for the after-hours veterinary intake flow. The customer lookup flow is implemented, while the AI agent, RAG layer, database, scheduling system, voice interface, and frontend remain future work.

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
LEGACY_CRM_BASE_URL=http://127.0.0.1:8001 LEGACY_CRM_API_KEY=dev-crm-key SCHEDULER_BASE_URL=http://127.0.0.1:8002 SCHEDULER_JWT_SECRET=dev-scheduler-secret HANDOFF_BASE_URL=http://127.0.0.1:8003 HANDOFF_WEBHOOK_SECRET=dev-handoff-secret uv run uvicorn app.main:app --reload --port 8000
```

Terminal 2: mock CRM service

```bash
MOCK_CRM_API_KEY=dev-crm-key uv run uvicorn mock_services.legacy_crm_api:app --reload --port 8001
```

Terminal 3: mock scheduling service

```bash
MOCK_SCHEDULER_JWT_SECRET=dev-scheduler-secret uv run uvicorn mock_services.scheduling_api:app --reload --port 8002
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

This checkpoint adds the first step of a small administrative policy knowledge layer for PawLine.

- The RAG system is only for administrative clinic policies, not veterinary advice or medical triage.
- PawLine safety routing remains the deterministic Python decision path for urgent medical concerns.
- Part 5A currently loads and chunks fictional clinic policy Markdown documents.
- Embedding, retrieval, and grounded answer generation belong to later checkpoints.

## Notes

- Python 3.12+ is targeted for the project.
- Dependencies are managed with `uv` instead of `pip`.
- The mock CRM is intentionally separate from the PawLine app and is only used for local development.
