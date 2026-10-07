"""Route one caller message to the appropriate specialist without answering it."""

from __future__ import annotations

import inspect
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field


class RouterRequestError(RuntimeError):
    """Raised when the router's OpenAI request cannot be completed."""


class RouterResponseError(RuntimeError):
    """Raised when the router response is missing or cannot be validated."""


class RouteDecision(BaseModel):
    """Validate the destination selected by the LLM for one caller message.

    The model returns untrusted structured data. This model checks that the
    selected destination is one of the five allowed specialist choices and that
    the internal reason contains usable text. The reason is for debugging only
    and must not be shown directly to the caller.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    destination: Literal["verification", "intake", "policy", "scheduling", "human_handoff"]
    reason: str = Field(..., min_length=1)


class OpenAIRouter:
    """Call OpenAI with router instructions and return a validated destination."""

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-6-luna") -> None:
        """Save the OpenAI client and model used by this dedicated router."""
        self._client = client
        self._model = model

    async def route(
        self,
        *,
        instructions: str,
        user_message: str,
    ) -> RouteDecision:
        """Return the single specialist destination chosen for the caller message.

        The caller supplies rendered router instructions and the current user
        message. This method does not render prompts, load conversation state,
        call business tools, or execute a specialist.
        """
        if not instructions or not instructions.strip():
            raise RouterRequestError("Router instructions cannot be blank.")
        if not user_message or not user_message.strip():
            raise RouterRequestError("User message cannot be blank.")

        try:
            parse_result = self._client.responses.parse( #response.parse is used when we require 1 specific result/ text_format
                model=self._model,
                input=user_message,
                instructions=instructions,
                text_format=RouteDecision, #structure the response like the Pydantic RouteDecision model.
            )
            if inspect.isawaitable(parse_result):
                response = await parse_result
            else: #normally the OpenAI call is asynchronous and must be awaited. this supports simple synchronous fake clients used in tests.
                response = parse_result
        except Exception as exc:
            raise RouterRequestError("Router request failed.") from exc #request problem

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RouterResponseError("Router response is missing parsed output.") #response problem, succeeded but empty

        try:
            if hasattr(parsed, "model_dump") and callable(parsed.model_dump): #pydantic object, convert to dict
                payload_data = parsed.model_dump()
            elif isinstance(parsed, dict): #already a dict, use it directly
                payload_data = parsed
            else:
                payload_data = getattr(parsed, "__dict__", {}) #fallback to __dict__ if available, otherwise empty dict
            return RouteDecision.model_validate(payload_data)
        except Exception as exc:
            raise RouterResponseError("Router response was malformed.") from exc


__all__ = ["OpenAIRouter", "RouteDecision", "RouterRequestError", "RouterResponseError"]
