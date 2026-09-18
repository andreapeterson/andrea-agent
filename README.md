# PawLine

PawLine is a minimal scaffold for an after-hours veterinary intake and escalation AI agent.

This repository intentionally contains only the initial project skeleton. The AI agent, RAG layer, database, customer system, scheduling system, voice interface, and frontend are not yet implemented.

## Local setup

This project uses `uv` as the package manager and runner.

```bash
cd /Users/andreapeterson/Documents/projects/pawline
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

## Project status

This scaffold provides the starting point for the PawLine service and includes a working health endpoint and test coverage for that endpoint.

## Notes

- Python 3.12+ is targeted for the project.
- Dependencies are managed with `uv` instead of `pip`.
- No AI agent, database, scheduling system, or frontend logic is included yet.
