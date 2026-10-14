"""Collect intake answers conversationally using the existing tool loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.record_intake_tool import RECORD_INTAKE_TOOL
from app.agents.request_handoff_tool import REQUEST_HANDOFF_TOOL
from app.agents.responses_tool_loop import run_agent_turn


class IntakeAgent:
    """Run intake instructions with a recording handler prepared for this conversation."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        prompt_renderer: PromptRenderer,
        record_intake_handler: Callable[[dict[str, object]], Awaitable[object]],
        request_handoff_handler: Callable[[dict[str, object]], Awaitable[object]],
    ) -> None:
        self._client = client
        self._model = model
        self._prompt_renderer = prompt_renderer
        self._record_intake_handler = record_intake_handler
        self._request_handoff_handler = request_handoff_handler

    async def respond(self, *, context: PromptContext, user_message: str) -> str:
        """Render the intake prompt, record requested facts, and return caller-facing text."""
        instructions = self._prompt_renderer.render_intake(context, user_message)
        # Connect the model's tool name to the prepared Python recording function.
        tool_handlers = {
            "record_intake": self._record_intake_handler,
            "request_handoff": self._request_handoff_handler,
        }

        # The shared loop calls OpenAI, executes requested tools, and sends their results back.
        return await run_agent_turn(
            client=self._client,
            model=self._model,
            instructions=instructions,
            user_message=user_message,
            tools=[RECORD_INTAKE_TOOL, REQUEST_HANDOFF_TOOL],
            tool_handlers=tool_handlers,
        )


__all__ = ["IntakeAgent"]
