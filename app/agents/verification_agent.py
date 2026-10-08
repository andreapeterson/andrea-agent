"""Talk with the caller to identify their customer account using the CRM lookup tool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.agents.lookup_customer_tool import LOOKUP_CUSTOMER_TOOL
from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.responses_tool_loop import run_agent_turn
from openai import AsyncOpenAI


class VerificationAgent:
    """Identify a caller conversationally and ask the CRM for a matching customer.

    It renders the verification prompt, supplies only the lookup_customer tool,
    and calls the existing Responses API tool loop. It never performs CRM
    authentication or decides routing.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        prompt_renderer: PromptRenderer,
        lookup_customer_handler: Callable[
            [dict[str, object]],
            Awaitable[object],
        ],
    ) -> None:
        self._client = client
        self._model = model
        self._prompt_renderer = prompt_renderer
        self._lookup_customer_handler = lookup_customer_handler

    async def respond(
        self,
        *,
        context: PromptContext,
        user_message: str,
    ) -> str:
        """Run one caller message through the verification prompt and CRM tool loop.

        This method runs after Python routing has selected the Verification Agent.
        The model may request lookup_customer, and the approved Python handler then
        contacts the CRM. The method returns the agent's caller-facing response.
        """
        instructions = self._prompt_renderer.render_verification(context, user_message)
        tool_handlers = {
            "lookup_customer": self._lookup_customer_handler,
        }

        return await run_agent_turn(
            client=self._client,
            model=self._model,
            instructions=instructions,
            user_message=user_message,
            tools=[LOOKUP_CUSTOMER_TOOL],
            tool_handlers=tool_handlers,
        )


__all__ = ["VerificationAgent"]
