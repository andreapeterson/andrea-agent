"""Request human assistance using the prepared handoff tool and shared model loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.request_handoff_tool import REQUEST_HANDOFF_TOOL
from app.agents.responses_tool_loop import run_agent_turn


class HumanHandoffAgent:
    """Run handoff instructions with the function prepared for this conversation."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        prompt_renderer: PromptRenderer,
        request_handoff_handler: Callable[[dict[str, object]], Awaitable[object]],
    ) -> None:
        self._client = client
        self._model = model
        self._prompt_renderer = prompt_renderer
        self._request_handoff_handler = request_handoff_handler

    async def respond(self, *, context: PromptContext, user_message: str) -> str:
        """Let the model request a handoff and describe the tool's actual result."""
        instructions = self._prompt_renderer.render_human_handoff(context, user_message)
        tool_handlers = {"request_handoff": self._request_handoff_handler}

        return await run_agent_turn(
            client=self._client,
            model=self._model,
            instructions=instructions,
            user_message=user_message,
            tools=[REQUEST_HANDOFF_TOOL],
            tool_handlers=tool_handlers,
        )


__all__ = ["HumanHandoffAgent"]
