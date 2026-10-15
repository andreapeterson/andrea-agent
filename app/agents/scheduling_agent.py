"""Find, select, and explicitly confirm appointments using prepared handlers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.responses_tool_loop import run_agent_turn
from app.agents.scheduling_tools import BOOK_SELECTED_APPOINTMENT_TOOL, FIND_APPOINTMENT_SLOTS_TOOL, SELECT_APPOINTMENT_SLOT_TOOL


class SchedulingAgent:
    """Run scheduling instructions with this conversation's prepared scheduling tools."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        prompt_renderer: PromptRenderer,
        find_appointment_slots_handler: Callable[[dict[str, object]], Awaitable[object]],
        select_appointment_slot_handler: Callable[[dict[str, object]], Awaitable[object]],
        book_selected_appointment_handler: Callable[[dict[str, object]], Awaitable[object]],
    ) -> None:
        self._client = client
        self._model = model
        self._prompt_renderer = prompt_renderer
        self._find_appointment_slots_handler = find_appointment_slots_handler
        self._select_appointment_slot_handler = select_appointment_slot_handler
        self._book_selected_appointment_handler = book_selected_appointment_handler

    async def respond(self, *, context: PromptContext, user_message: str) -> str:
        """Render the prompt; the shared loop executes tools and returns caller-facing text."""
        instructions = self._prompt_renderer.render_scheduling(context, user_message)
        tool_handlers = {
            "find_appointment_slots": self._find_appointment_slots_handler,
            "select_appointment_slot": self._select_appointment_slot_handler,
            "book_selected_appointment": self._book_selected_appointment_handler,
        }
        return await run_agent_turn(
            client=self._client,
            model=self._model,
            instructions=instructions,
            user_message=user_message,
            tools=[FIND_APPOINTMENT_SLOTS_TOOL, SELECT_APPOINTMENT_SLOT_TOOL, BOOK_SELECTED_APPOINTMENT_TOOL],
            tool_handlers=tool_handlers,
        )


__all__ = ["SchedulingAgent"]
