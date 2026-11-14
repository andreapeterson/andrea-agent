"""Evaluate deterministic request budgets using fake model responses and handlers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.agents import responses_tool_loop
from app.agents.resource_limits import (
    MAX_INSTRUCTION_BYTES,
    MAX_TOOL_ARGUMENT_BYTES,
    MAX_TOOL_RESULT_BYTES,
    MAX_TOOL_ROUNDS,
    MAX_USER_MESSAGE_CHARS,
    CallerMessageTooLargeError,
    InstructionsTooLargeError,
    ToolArgumentsTooLargeError,
    ToolResultTooLargeError,
)
from app.agents.responses_tool_loop import (
    MalformedToolCallError,
    ToolLoopLimitError,
    run_agent_turn,
)
from app.agents.router_agent import OpenAIRouter, RouteDecision


def function_call(arguments: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(type="function_call", call_id="call-demo", name="demo_tool", arguments=arguments)


def model_output(*calls: object, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(output=list(calls), output_text=text)


def fake_client(outputs: list[object]) -> SimpleNamespace:
    return SimpleNamespace(responses=SimpleNamespace(
        create=AsyncMock(side_effect=outputs),
        parse=AsyncMock(return_value=SimpleNamespace(
            output_parsed=RouteDecision(destination="verification", reason="Demo route."),
        )),
    ))


async def run_turn(client, handler, **overrides) -> str:
    parameters = {
        "client": client, "model": "demo-model", "instructions": "Demo instructions.",
        "user_message": "Please help.", "tools": [], "tool_handlers": {"demo_tool": handler},
    }
    parameters.update(overrides)
    return await run_agent_turn(**parameters)


def sized_json(byte_count: int, character: str) -> str:
    """Build valid JSON of an exact byte size, including its JSON punctuation."""
    prefix = "PRIVATE_TOOL_PAYLOAD_MARKER:"
    overhead = len(json.dumps({"text": prefix}, ensure_ascii=False).encode("utf-8"))
    count, remainder = divmod(byte_count - overhead, len(character.encode("utf-8")))
    result = json.dumps({"text": prefix + character * count + "x" * remainder}, ensure_ascii=False)
    assert len(result.encode("utf-8")) == byte_count
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["router", "tool_loop"])
async def test_oversized_caller_input_makes_no_model_or_handler_calls(boundary) -> None:
    client = fake_client([])
    handler = AsyncMock()
    rejected = "PRIVATE_CALLER_MARKER" + "x" * MAX_USER_MESSAGE_CHARS

    with pytest.raises(CallerMessageTooLargeError) as error:
        if boundary == "router":
            await OpenAIRouter(client).route(instructions="Demo instructions.", user_message=rejected)
        else:
            await run_turn(client, handler, user_message=rejected)

    client.responses.create.assert_not_awaited()
    client.responses.parse.assert_not_awaited()
    handler.assert_not_awaited()
    assert "PRIVATE_CALLER_MARKER" not in str(error.value)
    assert rejected not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["router", "tool_loop"])
async def test_caller_limit_counts_characters_and_accepts_exact_limit(boundary) -> None:
    client = fake_client([model_output(text="Hello.")])
    message = "🐕" * MAX_USER_MESSAGE_CHARS

    if boundary == "router":
        await OpenAIRouter(client).route(instructions="Demo instructions.", user_message=message)
        assert client.responses.parse.await_args.kwargs["input"] == message
    else:
        assert await run_turn(client, AsyncMock(), user_message=message) == "Hello."
        assert client.responses.create.await_args.kwargs["input"][0]["content"] == message


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["router", "tool_loop"])
async def test_oversized_rendered_context_makes_no_model_calls(boundary) -> None:
    client = fake_client([])
    instructions = "PRIVATE_HISTORY_MARKER" + "🐕" * (MAX_INSTRUCTION_BYTES // 4)
    assert len(instructions) < MAX_INSTRUCTION_BYTES

    with pytest.raises(InstructionsTooLargeError) as error:
        if boundary == "router":
            await OpenAIRouter(client).route(instructions=instructions, user_message="Hello.")
        else:
            await run_turn(client, AsyncMock(), instructions=instructions)

    client.responses.create.assert_not_awaited()
    client.responses.parse.assert_not_awaited()
    assert "PRIVATE_HISTORY_MARKER" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("character", ["a", "é", "🐕"])
async def test_oversized_arguments_are_rejected_before_json_parsing_or_handler(monkeypatch, character) -> None:
    raw = sized_json(MAX_TOOL_ARGUMENT_BYTES + 1, character)
    client = fake_client([model_output(function_call(raw))])
    handler = AsyncMock()
    parse = Mock(side_effect=AssertionError("Oversized JSON must not be parsed."))
    monkeypatch.setattr(responses_tool_loop.json, "loads", parse)

    with pytest.raises(ToolArgumentsTooLargeError) as error:
        await run_turn(client, handler)

    assert client.responses.create.await_count == 1
    parse.assert_not_called()
    handler.assert_not_awaited()
    assert raw not in str(error.value)
    assert "PRIVATE_TOOL_PAYLOAD_MARKER" not in str(error.value)
    assert "text" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("character", ["a", "é", "🐕"])
async def test_argument_at_exact_byte_limit_reaches_handler_without_truncation(character) -> None:
    raw = sized_json(MAX_TOOL_ARGUMENT_BYTES, character)
    client = fake_client([model_output(function_call(raw)), model_output(text="Done.")])
    handler = AsyncMock(return_value={"ok": True})

    assert await run_turn(client, handler) == "Done."

    handler.assert_awaited_once_with(json.loads(raw))


@pytest.mark.asyncio
@pytest.mark.parametrize("character", ["a", "é", "🐕"])
async def test_oversized_result_stops_before_function_output_and_second_request(character) -> None:
    result = json.loads(sized_json(MAX_TOOL_RESULT_BYTES + 1, character))
    client = fake_client([model_output(function_call())])
    handler = AsyncMock(return_value=result)

    with pytest.raises(ToolResultTooLargeError) as error:
        await run_turn(client, handler)

    handler.assert_awaited_once_with({})
    assert client.responses.create.await_count == 1
    # AsyncMock retains the input list, so this also checks nothing was appended.
    assert not any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in client.responses.create.await_args.kwargs["input"]
    )
    assert result["text"] not in str(error.value)
    assert "PRIVATE_TOOL_PAYLOAD_MARKER" not in str(error.value)
    assert "text" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("character", ["a", "é", "🐕"])
async def test_result_at_exact_byte_limit_is_serialized_once_and_sent_whole(monkeypatch, character) -> None:
    raw = sized_json(MAX_TOOL_RESULT_BYTES, character)
    handler = AsyncMock(return_value=json.loads(raw))
    client = fake_client([model_output(function_call()), model_output(text="Done.")])
    serialize = Mock(wraps=json.dumps)
    monkeypatch.setattr(responses_tool_loop.json, "dumps", serialize)

    assert await run_turn(client, handler) == "Done."

    serialize.assert_called_once_with(handler.return_value, ensure_ascii=False, allow_nan=False)
    output = client.responses.create.await_args.kwargs["input"][-1]
    assert output["type"] == "function_call_output"
    assert output["output"] == raw


@pytest.mark.asyncio
async def test_exactly_allowed_tool_rounds_can_finish_with_final_speech() -> None:
    client = fake_client([model_output(function_call()) for _ in range(MAX_TOOL_ROUNDS)] + [model_output(text="Done.")])
    handler = AsyncMock(return_value={"ok": True})

    assert await run_turn(client, handler) == "Done."

    assert handler.await_count == MAX_TOOL_ROUNDS
    assert client.responses.create.await_count == MAX_TOOL_ROUNDS + 1
    assert all(call.kwargs["parallel_tool_calls"] is False for call in client.responses.create.await_args_list)


@pytest.mark.asyncio
async def test_one_more_requested_tool_round_stops_before_another_handler() -> None:
    client = fake_client([model_output(function_call()) for _ in range(MAX_TOOL_ROUNDS + 1)])
    handler = AsyncMock(return_value={"ok": True})

    with pytest.raises(ToolLoopLimitError, match=f"limit of {MAX_TOOL_ROUNDS}"):
        await run_turn(client, handler)

    assert handler.await_count == MAX_TOOL_ROUNDS
    assert client.responses.create.await_count == MAX_TOOL_ROUNDS + 1
    assert all(call.kwargs["parallel_tool_calls"] is False for call in client.responses.create.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [MAX_TOOL_ROUNDS + 1, -1, True, 1.5])
async def test_internal_callers_cannot_raise_or_invalidate_tool_round_budget(limit) -> None:
    client = fake_client([])
    handler = AsyncMock()

    with pytest.raises(ValueError, match="max_tool_rounds must be an integer"):
        await run_turn(client, handler, max_tool_rounds=limit)

    client.responses.create.assert_not_awaited()
    handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_parallel_function_calls_still_fail_without_any_handler_execution() -> None:
    client = fake_client([model_output(function_call(), function_call())])
    handler = AsyncMock()

    with pytest.raises(MalformedToolCallError, match="more than one function call"):
        await run_turn(client, handler)

    handler.assert_not_awaited()
    assert client.responses.create.await_count == 1
    assert client.responses.create.await_args.kwargs["parallel_tool_calls"] is False
