"""Test the first real router-to-verification handoff."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest

from app.agents import (
    LOOKUP_CUSTOMER_TOOL,
    OpenAIRouter,
    PromptContext,
    PromptRenderer,
    RouteDecision,
    RouterRequestError,
)
from app.agents.lookup_customer_tool import create_lookup_customer_handler
from app.agents.responses_tool_loop import AgentModelRequestError, ToolExecutionError
from app.agents.router_agent import RouterResponseError
from app.agents.router_handoff import SpecialistNotAvailableError, route_and_respond
from app.agents.verification_agent import VerificationAgent
from app.models.customer import Customer, Pet, PetSpecies


class FakeLegacyCRMClient:
    def __init__(self, result: Customer | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[str] = []

    async def find_customer_by_phone(self, phone: str) -> Customer | None:
        self.calls.append(phone)
        if self.error is not None:
            raise self.error
        return self.result


class FakeOpenAIResponse:
    def __init__(self, payload: object | None = None, *, output: list[object] | None = None, output_text: str | None = None) -> None:
        self.output_parsed = payload
        self.output = output
        self.output_text = output_text


class FakeOpenAIResponses:
    def __init__(self, *, parsed: object | None = None, responses: list[object] | None = None, request_error: Exception | None = None) -> None:
        self.parsed = parsed
        self.responses = list(responses or [])
        self.request_error = request_error
        self.parse_calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> FakeOpenAIResponse:
        self.parse_calls.append(dict(kwargs))
        if self.request_error is not None:
            raise self.request_error
        return FakeOpenAIResponse(self.parsed)

    async def create(self, **kwargs: object) -> object:
        self.create_calls.append(dict(kwargs))
        if self.request_error is not None:
            raise self.request_error
        return self.responses.pop(0)


class FakeOpenAIClient:
    def __init__(self, responses: FakeOpenAIResponses) -> None:
        self.responses = responses


def make_context() -> PromptContext:
    return PromptContext(
        active_agent="verification",
        customer_verified=False,
        recent_messages=[
            {"role": "user", "content": "I need help."},
        ],
    )


def make_customer() -> Customer:
    return Customer(
        customer_id="customer-1001",
        first_name="Morgan",
        last_name="Example",
        phone_number="321-555-0100",
        pets=[
            Pet(pet_id="pet-1", name="Milo", species=PetSpecies.DOG),
            Pet(pet_id="pet-2", name="Pip", species=PetSpecies.CAT),
        ],
    )


def function_call(call_id: str = "call-customer-1", phone: str = "321-555-0100") -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name="lookup_customer",
        arguments=json.dumps({"phone": phone}),
    )


@pytest.mark.asyncio
async def test_router_destination_verification_invokes_verification_agent() -> None:
    router_client = FakeOpenAIClient(
        FakeOpenAIResponses(parsed=RouteDecision(destination="verification", reason="Caller supplied a phone number."))
    )
    verification_client = FakeOpenAIClient(
        FakeOpenAIResponses(
            responses=[
                FakeOpenAIResponse(
                    output=[function_call()],
                    output_text="",
                ),
                FakeOpenAIResponse(
                    output=[],
                    output_text="I found Morgan's account.",
                ),
            ]
        )
    )
    crm_client = FakeLegacyCRMClient(result=make_customer())
    agent = VerificationAgent(
        client=verification_client,
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        crm_client=crm_client,
    )
    service = route_and_respond(
        router=OpenAIRouter(router_client, model="router-model"),
        renderer=PromptRenderer(),
        verification_agent=agent,
    )

    result = await service(
        context=make_context(),
        user_message="My phone number is 321-555-0100.",
    )

    assert result == "I found Morgan's account."
    assert router_client.responses.parse_calls[0]["model"] == "router-model"
    assert router_client.responses.parse_calls[0]["input"] == "My phone number is 321-555-0100."
    assert verification_client.responses.create_calls[0]["tools"] == [LOOKUP_CUSTOMER_TOOL]
    assert crm_client.calls == ["321-555-0100"]
    assert len(verification_client.responses.create_calls) == 2


@pytest.mark.asyncio
async def test_verification_agent_uses_existing_lookup_customer_handler() -> None:
    crm_client = FakeLegacyCRMClient(result=make_customer())
    client = FakeOpenAIClient(
        FakeOpenAIResponses(
            responses=[
                FakeOpenAIResponse(
                    output=[function_call()],
                    output_text="",
                ),
                FakeOpenAIResponse(
                    output=[],
                    output_text="Milo and Pip are on the account.",
                ),
            ]
        )
    )
    agent = VerificationAgent(
        client=client,
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        crm_client=crm_client,
    )

    result = await agent.respond(
        context=make_context(),
        user_message="My phone number is 321-555-0100.",
    )

    assert result == "Milo and Pip are on the account."
    assert client.responses.create_calls[0]["tools"] == [LOOKUP_CUSTOMER_TOOL]
    second_input = cast(list[dict[str, object]], client.responses.create_calls[1]["input"])
    tool_output = cast(dict[str, object], second_input[2])
    assert tool_output["call_id"] == "call-customer-1"
    assert json.loads(cast(str, tool_output["output"])) == {
        "found": True,
        "first_name": "Morgan",
        "pets": [
            {"pet_id": "pet-1", "name": "Milo"},
            {"pet_id": "pet-2", "name": "Pip"},
        ],
    }


@pytest.mark.asyncio
async def test_customer_not_found_produces_safe_response() -> None:
    crm_client = FakeLegacyCRMClient(result=None)
    client = FakeOpenAIClient(
        FakeOpenAIResponses(
            responses=[
                FakeOpenAIResponse(
                    output=[function_call(phone="999-999-9999")],
                    output_text="",
                ),
                FakeOpenAIResponse(
                    output=[],
                    output_text="I could not find an account with that phone number.",
                ),
            ]
        )
    )
    agent = VerificationAgent(
        client=client,
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        crm_client=crm_client,
    )

    result = await agent.respond(
        context=make_context(),
        user_message="My phone number is 999-999-9999.",
    )

    assert result == "I could not find an account with that phone number."
    assert crm_client.calls == ["999-999-9999"]


@pytest.mark.asyncio
async def test_unimplemented_destination_raises_specialist_error() -> None:
    router_client = FakeOpenAIClient(
        FakeOpenAIResponses(parsed=RouteDecision(destination="policy", reason="General clinic information."))
    )
    verification_client = FakeOpenAIClient(FakeOpenAIResponses(responses=[]))
    service = route_and_respond(
        router=OpenAIRouter(router_client, model="router-model"),
        renderer=PromptRenderer(),
        verification_agent=VerificationAgent(
            client=verification_client,
            model="verification-model",
            prompt_renderer=PromptRenderer(),
            crm_client=FakeLegacyCRMClient(),
        ),
    )

    with pytest.raises(SpecialistNotAvailableError, match="policy"):
        await service(context=make_context(), user_message="What are your hours?")


@pytest.mark.asyncio
async def test_router_failure_does_not_invoke_verification_agent() -> None:
    router_client = FakeOpenAIClient(FakeOpenAIResponses(request_error=RuntimeError("private router detail")))
    verification_client = FakeOpenAIClient(FakeOpenAIResponses(responses=[]))
    service = route_and_respond(
        router=OpenAIRouter(router_client, model="router-model"),
        renderer=PromptRenderer(),
        verification_agent=VerificationAgent(
            client=verification_client,
            model="verification-model",
            prompt_renderer=PromptRenderer(),
            crm_client=FakeLegacyCRMClient(),
        ),
    )

    with pytest.raises(RouterRequestError, match="Router request failed") as error:
        await service(context=make_context(), user_message="My phone number is 321-555-0100.")

    assert "private router detail" not in str(error.value)
    assert verification_client.responses.create_calls == []


@pytest.mark.asyncio
async def test_verification_failure_is_not_reported_as_router_failure() -> None:
    router_client = FakeOpenAIClient(
        FakeOpenAIResponses(parsed=RouteDecision(destination="verification", reason="Caller supplied a phone number."))
    )
    verification_client = FakeOpenAIClient(
        FakeOpenAIResponses(
            request_error=RuntimeError("private verification detail"),
        )
    )
    service = route_and_respond(
        router=OpenAIRouter(router_client, model="router-model"),
        renderer=PromptRenderer(),
        verification_agent=VerificationAgent(
            client=verification_client,
            model="verification-model",
            prompt_renderer=PromptRenderer(),
            crm_client=FakeLegacyCRMClient(),
        ),
    )

    with pytest.raises(AgentModelRequestError, match="Responses API request failed"):
        await service(context=make_context(), user_message="My phone number is 321-555-0100.")

    assert len(verification_client.responses.create_calls) == 1
