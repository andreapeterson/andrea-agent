"""Run one caller message directly through the OpenAI Responses API tool loop."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable

from openai import AsyncOpenAI


class AgentModelRequestError(RuntimeError):
    """Raised when an OpenAI Responses API request fails."""


class UnknownAgentToolError(RuntimeError):
    """Raised when the model requests a tool without an approved handler."""


class MalformedToolCallError(RuntimeError):
    """Raised when a function call is missing fields or has invalid arguments."""


class ToolExecutionError(RuntimeError):
    """Raised when an approved Python tool fails or its result is not JSON-safe."""


class ToolLoopLimitError(RuntimeError):
    """Raised when the model requests more tools than this caller turn permits."""


async def run_agent_turn(
    *,
    client: AsyncOpenAI,
    model: str,
    instructions: str,
    user_message: str,
    tools: list[dict[str, object]], #describes available tools 
    tool_handlers: dict[str, Callable[[dict[str, object]], object]], #map from tool names to python functions
    max_tool_rounds: int = 3, #one caller message can only call 3 tools max
) -> str:
    """Ask the model to answer one caller message, running approved tools as needed."""
    if not instructions or not instructions.strip():
        raise ValueError("Instructions cannot be blank.")
    if not user_message or not user_message.strip():
        raise ValueError("Caller message cannot be blank.")
    if max_tool_rounds < 0:
        raise ValueError("max_tool_rounds cannot be negative.")

    input_items: list[object] = [{"role": "user", "content": user_message}] #running history during this turn
    tool_rounds = 0

    while True:
        # First model request: the model either answers or asks for a tool.
        try:
            response = await client.responses.create(
                model=model,
                instructions=instructions,
                input=input_items,
                tools=tools,
                parallel_tool_calls=False, #model can only request 1 tool at a time
            )
        except Exception as exc:
            raise AgentModelRequestError("The OpenAI Responses API request failed.") from exc

        output_items = getattr(response, "output", None)
        if not isinstance(output_items, list):
            raise ValueError("The model response did not contain an output list.")

        function_calls = [item for item in output_items if getattr(item, "type", None) == "function_call"]
        if len(function_calls) > 1:
            raise MalformedToolCallError("The model returned more than one function call.")

        if not function_calls:
            final_text = getattr(response, "output_text", None)
            if not isinstance(final_text, str) or not final_text.strip():
                raise ValueError("The model returned neither a function call nor nonblank assistant text.")
            return final_text

        if tool_rounds >= max_tool_rounds:
            raise ToolLoopLimitError(f"The model exceeded the limit of {max_tool_rounds} tool rounds.")

        function_call = function_calls[0]
        call_id = getattr(function_call, "call_id", None) #call id connects models request to the tool result
        tool_name = getattr(function_call, "name", None)
        raw_arguments = getattr(function_call, "arguments", None)
        if not isinstance(call_id, str) or not call_id.strip():
            raise MalformedToolCallError("The function call did not include a call_id.")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise MalformedToolCallError("The function call did not include a tool name.")
        if not isinstance(raw_arguments, str):
            raise MalformedToolCallError("The function call arguments were not JSON text.")

        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            raise MalformedToolCallError("The function call arguments were invalid JSON.") from exc
        if not isinstance(arguments, dict):
            raise MalformedToolCallError("The function call arguments must be a JSON object.")

        handler = tool_handlers.get(tool_name)
        if handler is None:
            raise UnknownAgentToolError(f"The model requested an unapproved tool: {tool_name}.")

        # Preserve the model's function call for the next API request.
        input_items.extend(output_items)

        # Python executes the approved tool; the LLM does not run it itself.
        try:
            tool_result = handler(arguments)
            if inspect.isawaitable(tool_result):
                tool_result = await tool_result
            serialized_result = json.dumps(tool_result, ensure_ascii=False, allow_nan=False)
        except Exception as exc:
            raise ToolExecutionError("The approved tool failed or returned a non-JSON result.") from exc

        # Match the tool result to the model's request using call_id.
        input_items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": serialized_result,
            }
        )
        tool_rounds += 1

        # Ask the model again so it can turn the tool result into a caller response.