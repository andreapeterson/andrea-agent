# PawLine

PawLine is a minimal scaffold for an after-hours veterinary intake and escalation AI agent.

This repository intentionally contains only the initial project skeleton. The AI agent, RAG layer, database, customer system, scheduling system, voice interface, and frontend are not yet implemented.

## Local setup

This project uses `uv` as the package manager and runner.

```bash
cd pawline
uv sync --dev
uv run pytest -q
uv run uvicorn app.main:app --reload
```

## Health check

Once the app is running:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status": "ok"}
```

## Mock CRM service

The repository also includes a separate mock legacy CRM service for local development only.

Run it on port 8001:

```bash
cd pawline
uv run uvicorn mock_services.legacy_crm_api:app --host 127.0.0.1 --port 8001 --reload
```

Example request:

```bash
curl -H "X-API-Key: dev-crm-key" "http://127.0.0.1:8001/customers/by-phone?phone=3215550100"
```

## Notes

- Python 3.12+ is targeted for the project.
- Dependencies are managed with `uv` instead of `pip`.
- The mock CRM is intentionally separate from the PawLine app and is not yet connected to the customer lookup flow.
