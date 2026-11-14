"""Select one Router at construction time, shared by runtime and live Router eval."""

from typing import Literal, cast

from openai import AsyncOpenAI

from app.agents.decisions_router import OpenAIDecisionsRouter
from app.agents.router_agent import OpenAIRouter, Router

RouterBackend = Literal["responses", "decisions"]


def validate_router_backend(backend: str) -> RouterBackend:
    if backend not in {"responses", "decisions"}:
        raise ValueError("PAWLINE_ROUTER_BACKEND must be responses or decisions.")
    return cast(RouterBackend, backend)


def create_router(
    client: AsyncOpenAI, *, backend: RouterBackend = "responses", model: str = "gpt-6-luna",
) -> Router:
    """The model setting applies to Responses; Decisions uses its supported model."""
    validate_router_backend(backend)
    if backend == "decisions":
        return OpenAIDecisionsRouter(client)
    return OpenAIRouter(client, model=model)
