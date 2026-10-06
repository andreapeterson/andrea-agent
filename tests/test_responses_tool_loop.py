"""Test the Responses API tool loop with a fake client and local handlers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

from app.agents import (
    AgentModelRequestError,
    MalformedToolCallError,
    ToolExecutionError,
    ToolLoopLimitError,
    UnknownAgentToolError,
    run_agent_turn,
)


def make_tool() -> dict[str, object]:
    """Return the test-only function schema sent to the model."""
    return {
        "type": "function",
        "name": "lookup_weather",
        "description": "Return the weather for one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def make_function_call(
    *,
    call_id: str = "call-weather-1",
    name: str = "lookup_weather",
    arguments: str = '{"city":"Oslo"}',
) -> SimpleNamespace:
    """Make an SDK-shaped function call item for a fake model response."""
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


class FakeResponses:
    """Return prepared API responses and snapshot each request's input list."""

    def __init__(self, responses: list[object] | None = None, error: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        request = dict(kwargs)
        request["input"] = list(cast(list[object], kwargs["input"]))
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


def fake_client(responses: FakeResponses) -> AsyncOpenAI:
    """Provide the one AsyncOpenAI attribute used by run_agent_turn."""
    return cast(AsyncOpenAI, SimpleNamespace(responses=responses))


@pytest.mark.asyncio
async def test_direct_answer_returns_text_without_running_a_handler() -> None:
    responses = FakeResponses([SimpleNamespace(output=[], output_text="Hello, Avery.")])
    handler = AsyncMock()

    result = await run_agent_turn(
        client=fake_client(responses),
        model="test-model",
        instructions="Front desk instructions",
        user_message="Hello.",
        tools=[make_tool()],
        tool_handlers={"lookup_weather": handler},
    )

    assert result == "Hello, Avery."
    assert handler.await_count == 0
    assert len(responses.requests) == 1
    assert responses.requests[0]["parallel_tool_calls"] is False
    assert responses.requests[0]["tools"] == [make_tool()]


@pytest.mark.asyncio
async def test_tool_call_runs_handler_and_sends_matching_result_back() -> None:
    original_function_call = make_function_call()
    responses = FakeResponses(
        [
            SimpleNamespace(output=[original_function_call], output_text=""),
            SimpleNamespace(output=[], output_text="It is cloudy in Oslo."),
        ]
    )
    handler = AsyncMock(return_value={"condition": "cloudy", "temperature_c": 12})

    result = await run_agent_turn(
        client=fake_client(responses),
        model="test-model",
        instructions="Front desk instructions",
        user_message="What is the weather in Oslo?",
        tools=[make_tool()],
        tool_handlers={"lookup_weather": handler},
    )

    assert result == "It is cloudy in Oslo."
    handler.assert_awaited_once_with({"city": "Oslo"})
    assert len(responses.requests) == 2
    second_input = cast(list[object], responses.requests[1]["input"])
    assert second_input[0] == {"role": "user", "content": "What is the weather in Oslo?"}
    assert second_input[1] is original_function_call
    tool_result = cast(dict[str, object], second_input[2])
    assert tool_result["type"] == "function_call_output"
    assert tool_result["call_id"] == original_function_call.call_id
    assert json.loads(cast(str, tool_result["output"])) == {
        "condition": "cloudy",
        "temperature_c": 12,
    }
    assert all(request["parallel_tool_calls"] is False for request in responses.requests)


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected_without_running_a_handler() -> None:
    responses = FakeResponses(
        [SimpleNamespace(output=[make_function_call(name="delete_records")], output_text="")]
    )
    handler = AsyncMock()

    with pytest.raises(UnknownAgentToolError, match="unapproved tool"):
        await run_agent_turn(
            client=fake_client(responses),
            model="test-model",
            instructions="instructions",
            user_message="do something",
            tools=[make_tool()],
            tool_handlers={"lookup_weather": handler},
        )

    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_function_arguments_are_rejected() -> None:
    responses = FakeResponses([SimpleNamespace(output=[make_function_call(arguments="not-json")], output_text="")])
    handler = AsyncMock()

    with pytest.raises(MalformedToolCallError, match="invalid JSON"):
        await run_agent_turn(
            client=fake_client(responses),
            model="test-model",
            instructions="instructions",
            user_message="check weather",
            tools=[make_tool()],
            tool_handlers={"lookup_weather": handler},
        )

    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_handler_failure_is_wrapped_without_claiming_success() -> None:
    responses = FakeResponses([SimpleNamespace(output=[make_function_call()], output_text="")])
    handler = AsyncMock(side_effect=RuntimeError("private failure detail"))

    with pytest.raises(ToolExecutionError, match="approved tool failed") as error:
        await run_agent_turn(
            client=fake_client(responses),
            model="test-model",
            instructions="instructions",
            user_message="check weather",
            tools=[make_tool()],
            tool_handlers={"lookup_weather": handler},
        )

    assert "private failure detail" not in str(error.value)


@pytest.mark.asyncio
async def test_tool_round_limit_stops_before_running_an_extra_tool() -> None:
    responses = FakeResponses(
        [
            SimpleNamespace(output=[make_function_call(call_id=f"call-{index}")], output_text="")
            for index in range(3)
        ]
    )
    handler = AsyncMock(return_value={"condition": "cloudy"})

    with pytest.raises(ToolLoopLimitError, match="limit of 2"):
        await run_agent_turn(
            client=fake_client(responses),
            model="test-model",
            instructions="instructions",
            user_message="check weather",
            tools=[make_tool()],
            tool_handlers={"lookup_weather": handler},
            max_tool_rounds=2,
        )

    assert len(responses.requests) == 3
    assert handler.await_count == 2


@pytest.mark.asyncio
async def test_blank_model_output_is_rejected() -> None:
    responses = FakeResponses([SimpleNamespace(output=[], output_text="  ")])

    with pytest.raises(ValueError, match="nonblank assistant text"):
        await run_agent_turn(
            client=fake_client(responses),
            model="test-model",
            instructions="instructions",
            user_message="hello",
            tools=[],
            tool_handlers={},
        )


@pytest.mark.asyncio
async def test_blank_inputs_and_api_failure_are_reported() -> None:
    responses = FakeResponses(error=RuntimeError("private API details"))

    with pytest.raises(ValueError, match="Instructions cannot be blank"):
        await run_agent_turn(
            client=fake_client(responses),
            model="test-model",
            instructions=" ",
            user_message="hello",
            tools=[],
            tool_handlers={},
        )

    with pytest.raises(ValueError, match="Caller message cannot be blank"):
        await run_agent_turn(
            client=fake_client(responses),
            model="test-model",
            instructions="instructions",
            user_message=" ",
            tools=[],
            tool_handlers={},
        )

    with pytest.raises(AgentModelRequestError, match="Responses API request failed") as error:
        await run_agent_turn(
            client=fake_client(responses),
            model="test-model",
            instructions="instructions",
            user_message="hello",
            tools=[],
            tool_handlers={},
        )

    assert "private API details" not in str(error.value)
    assert responses.requests[-1]["input"] == [{"role": "user", "content": "hello"}]