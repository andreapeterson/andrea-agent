"""Answer administrative clinic-policy questions using retrieved evidence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.responses_tool_loop import run_agent_turn
from app.agents.search_policy_tool import SEARCH_POLICY_TOOL


class PolicyAgent:
    """Run the policy prompt and search tool through the shared Responses loop."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        prompt_renderer: PromptRenderer,
        search_policy_handler: Callable[[dict[str, object]], Awaitable[object]],
    ) -> None:
        self._client = client
        self._model = model
        self._prompt_renderer = prompt_renderer
        self._search_policy_handler = search_policy_handler

    async def respond(self, *, context: PromptContext, user_message: str) -> str:
        """Render policy instructions and run one caller message through the tool loop.

        If the model requests search_policy, the loop runs the prepared handler
        and sends its evidence back to the model, then returns the caller-facing answer.
        """
        instructions = self._prompt_renderer.render_policy(context, user_message)
        tool_handlers = {"search_policy": self._search_policy_handler}

        return await run_agent_turn(
            client=self._client,
            model=self._model,
            instructions=instructions,
            user_message=user_message,
            tools=[SEARCH_POLICY_TOOL],
            tool_handlers=tool_handlers,
        )


__all__ = ["PolicyAgent"]
