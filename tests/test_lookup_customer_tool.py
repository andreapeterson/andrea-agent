"""Test the real customer lookup handler through a fake CRM and model client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from app.agents.lookup_customer_tool import (
    LOOKUP_CUSTOMER_TOOL,
    LookupCustomerArguments,
    create_lookup_customer_handler,
)
from app.agents.prompt_context import PromptContext
from app.agents.prompt_renderer import PromptRenderer
from app.agents.responses_tool_loop import ToolExecutionError, run_agent_turn
from app.models.customer import Customer, Pet, PetSpecies
from app.services.conversation_store import InMemoryConversationStore
from app.integrations.legacy_crm import LegacyCRMParseError, LegacyCRMRequestError


def make_customer() -> Customer:
    """Build an existing domain-model customer for fake CRM responses."""
    return Customer(
        customer_id="customer-private-id",
        first_name="Morgan",
        last_name="Example",
        phone_number="3215550100",
        pets=[
            Pet(pet_id="pet-1", name="Milo", species=PetSpecies.DOG),
            Pet(pet_id="pet-2", name="Pip", species=PetSpecies.CAT),
        ],
    )


class FakeLegacyCRMClient:
    """Return a prepared customer or error and record the phone supplied."""

    def __init__(self, result: Customer | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.phones: list[str] = []

    async def find_customer_by_phone(self, phone: str) -> Customer | None:
        self.phones.append(phone)
        if self.error is not None:
            raise self.error
        return self.result


class FakeResponses:
    """Capture Responses API requests and return prepared responses in order."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        request = dict(kwargs)
        request["input"] = list(cast(list[object], kwargs["input"]))
        self.requests.append(request)
        return self.responses.pop(0)


def make_fake_openai_client(responses: FakeResponses) -> AsyncOpenAI:
    """Provide only the client method used by run_agent_turn."""
    return cast(AsyncOpenAI, SimpleNamespace(responses=responses))


def function_call() -> SimpleNamespace:
    """Build the SDK-shaped function_call item emitted by the fake model."""
    return SimpleNamespace(
        type="function_call",
        call_id="call-customer-1",
        name="lookup_customer",
        arguments='{"phone":"(321) 555-0100"}',
    )


def test_lookup_arguments_accept_phone_and_reject_blank_or_extra_fields() -> None:
    valid = LookupCustomerArguments(phone=" +44 (20) 1234-5678 ")

    assert valid.phone == "+44 (20) 1234-5678"

    with pytest.raises(ValidationError):
        LookupCustomerArguments(phone="   ")
    with pytest.raises(ValidationError):
        LookupCustomerArguments(phone="3215550100", invented=True)

    parameters = cast(dict[str, object], LOOKUP_CUSTOMER_TOOL["parameters"])
    assert LOOKUP_CUSTOMER_TOOL["type"] == "function"
    assert LOOKUP_CUSTOMER_TOOL["name"] == "lookup_customer"
    assert LOOKUP_CUSTOMER_TOOL["strict"] is True
    assert parameters["required"] == ["phone"]
    assert parameters["additionalProperties"] is False


@pytest.mark.asyncio
async def test_lookup_handler_returns_only_approved_customer_details() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    crm_client = FakeLegacyCRMClient(result=make_customer())

    handler = create_lookup_customer_handler(crm_client, conversation_state=state, conversation_store=store)

    result = await handler({"phone": "(321) 555-0100"})

    assert crm_client.phones == ["(321) 555-0100"]
    assert result == {
        "found": True,
        "first_name": "Morgan",
        "pets": [
            {"pet_id": "pet-1", "name": "Milo"},
            {"pet_id": "pet-2", "name": "Pip"},
        ],
    }
    serialized = json.dumps(result)
    assert "321" not in serialized
    assert "customer-private-id" not in serialized
    assert "Example" not in serialized
    assert "species" not in serialized


@pytest.mark.asyncio
async def test_lookup_not_found_is_a_normal_tool_result() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    crm_client = FakeLegacyCRMClient(result=None)

    handler = create_lookup_customer_handler(crm_client, conversation_state=state, conversation_store=store)

    result = await handler({"phone": "9999999999"})

    assert crm_client.phones == ["9999999999"]
    assert result == {"found": False}


@pytest.mark.asyncio
async def test_lookup_handler_saves_customer_once_for_successful_lookup() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    crm_client = FakeLegacyCRMClient(result=make_customer())

    handler = create_lookup_customer_handler(crm_client, conversation_state=state, conversation_store=store)

    result = await handler({"phone": "(321) 555-0100"})

    assert result["found"] is True
    store.save.assert_called_once_with(state)
    assert store.get(state.conversation_id).verified_customer == make_customer()
    assert store.get(state.conversation_id).verified_customer_id == "customer-private-id"


@pytest.mark.asyncio
async def test_lookup_not_found_does_not_save_verified_customer() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    crm_client = FakeLegacyCRMClient(result=None)

    handler = create_lookup_customer_handler(crm_client, conversation_state=state, conversation_store=store)

    result = await handler({"phone": "9999999999"})

    assert result == {"found": False}
    store.save.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_crm_failure_does_not_save_verified_customer() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    crm_client = FakeLegacyCRMClient(error=LegacyCRMRequestError("private detail"))

    handler = create_lookup_customer_handler(crm_client, conversation_state=state, conversation_store=store)

    with pytest.raises(LegacyCRMRequestError):
        await handler({"phone": "3215550100"})
    store.save.assert_not_called()


@pytest.mark.asyncio
async def test_two_call_lookup_flow_uses_rendered_prompt_and_safe_tool_result() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    caller_message = "My phone number is (321) 555-0100. Which pets are on my account?"
    function_call_item = function_call()
    fake_responses = FakeResponses(
        [
            SimpleNamespace(output=[function_call_item], output_text=""),
            SimpleNamespace(output=[], output_text="I found Morgan's account with Milo and Pip."),
        ]
    )
    openai_client = make_fake_openai_client(fake_responses)
    crm_client = FakeLegacyCRMClient(result=make_customer())

    prompt_context = PromptContext(available_tools=["lookup_customer"])
    rendered_instructions = PromptRenderer().render_front_desk(prompt_context)

    # The tool definition tells the LLM that lookup_customer exists.
    # This dictionary connects that tool name to the Python function that
    # actually calls LegacyCRMClient.
    #aka the handler registry connects the LLM’s requested name to the real PawLine Python function
    tool_handlers = {
        "lookup_customer": create_lookup_customer_handler(crm_client, conversation_state=state, conversation_store=store),
    }

    result = await run_agent_turn(
        client=openai_client,
        model="test-model",
        instructions=rendered_instructions,
        user_message=caller_message,
        tools=[LOOKUP_CUSTOMER_TOOL],
        tool_handlers=tool_handlers,
    )

    assert result == "I found Morgan's account with Milo and Pip."
    assert crm_client.phones == ["(321) 555-0100"]
    assert len(fake_responses.requests) == 2
    assert fake_responses.requests[0]["instructions"] == rendered_instructions
    second_input = cast(list[object], fake_responses.requests[1]["input"])
    assert second_input[0] == {"role": "user", "content": caller_message}
    assert second_input[1] is function_call_item
    tool_output = cast(dict[str, object], second_input[2])
    assert tool_output["type"] == "function_call_output"
    assert tool_output["call_id"] == function_call_item.call_id
    assert json.loads(cast(str, tool_output["output"])) == {
        "found": True,
        "first_name": "Morgan",
        "pets": [
            {"pet_id": "pet-1", "name": "Milo"},
            {"pet_id": "pet-2", "name": "Pip"},
        ],
    }


@pytest.mark.parametrize(
    "crm_error",
    [
        LegacyCRMRequestError("private request detail"),
        LegacyCRMParseError("private XML detail"),
    ],
)
@pytest.mark.asyncio
async def test_crm_failures_are_not_returned_as_not_found_or_sent_to_model(crm_error: Exception) -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    fake_responses = FakeResponses([SimpleNamespace(output=[function_call()], output_text="")])
    crm_client = FakeLegacyCRMClient(error=crm_error)

    handler = create_lookup_customer_handler(crm_client, conversation_state=state, conversation_store=store)
    prompt_context = PromptContext(available_tools=["lookup_customer"])

    with pytest.raises(ToolExecutionError, match="approved tool failed") as error:
        await run_agent_turn(
            client=make_fake_openai_client(fake_responses),
            model="test-model",
            instructions=PromptRenderer().render_front_desk(prompt_context),
            user_message="Look up my account.",
            tools=[LOOKUP_CUSTOMER_TOOL],
            tool_handlers={"lookup_customer": handler},
        )

    assert "private" not in str(error.value)
    assert len(fake_responses.requests) == 1
    assert crm_client.phones == ["(321) 555-0100"]