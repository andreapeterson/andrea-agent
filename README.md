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

Terminal 1: mock CRM service

```bash
MOCK_CRM_API_KEY=dev-crm-key uv run uvicorn mock_services.legacy_crm_api:app --reload --port 8001
```

Terminal 2: PawLine app

```bash
LEGACY_CRM_BASE_URL=http://127.0.0.1:8001 LEGACY_CRM_API_KEY=dev-crm-key uv run uvicorn app.main:app --reload --port 8000
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

## Notes

- Python 3.12+ is targeted for the project.
- Dependencies are managed with `uv` instead of `pip`.
- The mock CRM is intentionally separate from the PawLine app and is only used for local development.
