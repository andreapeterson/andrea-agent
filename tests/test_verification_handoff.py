"""Test the first real router-to-verification handoff."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

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
from app.services.conversation_store import InMemoryConversationStore


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
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
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
    lookup_customer_handler = create_lookup_customer_handler(
        crm_client,
        conversation_state=state,
        conversation_store=store,
    )
    agent = VerificationAgent(
        client=verification_client,
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        lookup_customer_handler=lookup_customer_handler,
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
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
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
    lookup_customer_handler = create_lookup_customer_handler(
        crm_client,
        conversation_state=state,
        conversation_store=store,
    )
    agent = VerificationAgent(
        client=client,
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        lookup_customer_handler=lookup_customer_handler,
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
async def test_successful_lookup_persists_verified_customer_id() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-verified")
    store.save = Mock(wraps=store.save)
    crm_client = FakeLegacyCRMClient(result=make_customer())

    lookup_customer_handler = create_lookup_customer_handler(
        crm_client,
        conversation_state=state,
        conversation_store=store,
    )
    agent = VerificationAgent(
        client=FakeOpenAIClient(
            FakeOpenAIResponses(
                responses=[
                    FakeOpenAIResponse(output=[function_call()], output_text=""),
                    FakeOpenAIResponse(output=[], output_text="I found Morgan's account."),
                ]
            )
        ),
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        lookup_customer_handler=lookup_customer_handler,
    )

    result = await agent.respond(
        context=make_context(),
        user_message="My phone number is 321-555-0100.",
    )

    assert result == "I found Morgan's account."
    store.save.assert_called_once_with(state)
    assert store.get("conv-verified").verified_customer == make_customer()
    assert store.get("conv-verified").verified_customer_id == "customer-1001"


@pytest.mark.asyncio
async def test_prepared_lookup_handlers_keep_conversations_isolated() -> None:
    store = InMemoryConversationStore()
    customers = [
        make_customer(),
        make_customer().model_copy(
            update={
                "customer_id": "customer-2002",
                "first_name": "Taylor",
                "phone_number": "321-555-0200",
            }
        ),
    ]

    class YieldingCRMClient(FakeLegacyCRMClient):
        async def find_customer_by_phone(self, phone: str) -> Customer | None:
            await asyncio.sleep(0)
            return await super().find_customer_by_phone(phone)

    agents = []
    crm_clients = []
    for index, customer in enumerate(customers):
        state = store.create(f"conv-isolated-{index}")
        state.original_concern = f"Caller {index}'s concern"
        crm_client = YieldingCRMClient(result=customer)
        crm_clients.append(crm_client)
        lookup_customer_handler = create_lookup_customer_handler(
            crm_client,
            conversation_state=state,
            conversation_store=store,
        )
        agents.append(
            VerificationAgent(
                client=FakeOpenAIClient(
                    FakeOpenAIResponses(
                        responses=[
                            FakeOpenAIResponse(
                                output=[function_call(phone=customer.phone_number)],
                                output_text="",
                            ),
                            FakeOpenAIResponse(output=[], output_text=customer.first_name),
                        ]
                    )
                ),
                model="verification-model",
                prompt_renderer=PromptRenderer(),
                lookup_customer_handler=lookup_customer_handler,
            )
        )

    results = await asyncio.gather(
        *(
            agent.respond(
                context=make_context(),
                user_message=f"My phone number is {customer.phone_number}.",
            )
            for agent, customer in zip(agents, customers)
        )
    )

    assert results == ["Morgan", "Taylor"]
    for index, customer in enumerate(customers):
        saved = store.get(f"conv-isolated-{index}")
        assert saved.verified_customer == customer
        assert saved.verified_customer_id == customer.customer_id
        assert saved.original_concern == f"Caller {index}'s concern"
        assert crm_clients[index].calls == [customer.phone_number]


@pytest.mark.asyncio
async def test_customer_not_found_does_not_save_verified_customer() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-not-found")
    store.save = Mock(wraps=store.save)
    crm_client = FakeLegacyCRMClient(result=None)

    lookup_customer_handler = create_lookup_customer_handler(
        crm_client,
        conversation_state=state,
        conversation_store=store,
    )
    agent = VerificationAgent(
        client=FakeOpenAIClient(
            FakeOpenAIResponses(
                responses=[
                    FakeOpenAIResponse(output=[function_call(phone="999-999-9999")], output_text=""),
                    FakeOpenAIResponse(output=[], output_text="I could not find an account."),
                ]
            )
        ),
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        lookup_customer_handler=lookup_customer_handler,
    )

    result = await agent.respond(
        context=make_context(),
        user_message="My phone number is 999-999-9999.",
    )

    assert result == "I could not find an account."
    store.save.assert_not_called()
    assert store.get("conv-not-found").verified_customer_id is None


@pytest.mark.asyncio
async def test_crm_failure_does_not_save_verified_customer() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-crm-failure")
    store.save = Mock(wraps=store.save)
    crm_client = FakeLegacyCRMClient(error=RuntimeError("private CRM detail"))

    lookup_customer_handler = create_lookup_customer_handler(
        crm_client,
        conversation_state=state,
        conversation_store=store,
    )
    agent = VerificationAgent(
        client=FakeOpenAIClient(
            FakeOpenAIResponses(
                responses=[
                    FakeOpenAIResponse(output=[function_call()], output_text=""),
                ]
            )
        ),
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        lookup_customer_handler=lookup_customer_handler,
    )

    with pytest.raises(RuntimeError, match="approved tool failed") as error:
        await agent.respond(
            context=make_context(),
            user_message="My phone number is 321-555-0100.",
        )

    assert "private CRM detail" not in str(error.value)
    store.save.assert_not_called()
    assert store.get("conv-crm-failure").verified_customer_id is None


@pytest.mark.asyncio
async def test_customer_not_found_produces_safe_response() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
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
    lookup_customer_handler = create_lookup_customer_handler(
        crm_client,
        conversation_state=state,
        conversation_store=store,
    )
    agent = VerificationAgent(
        client=client,
        model="verification-model",
        prompt_renderer=PromptRenderer(),
        lookup_customer_handler=lookup_customer_handler,
    )

    result = await agent.respond(
        context=make_context(),
        user_message="My phone number is 999-999-9999.",
    )

    assert result == "I could not find an account with that phone number."
    assert crm_client.calls == ["999-999-9999"]


@pytest.mark.asyncio
async def test_unimplemented_destination_raises_specialist_error() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    router_client = FakeOpenAIClient(
        FakeOpenAIResponses(parsed=RouteDecision(destination="policy", reason="General clinic information."))
    )
    verification_client = FakeOpenAIClient(FakeOpenAIResponses(responses=[]))
    lookup_customer_handler = create_lookup_customer_handler(
        FakeLegacyCRMClient(),
        conversation_state=state,
        conversation_store=store,
    )
    service = route_and_respond(
        router=OpenAIRouter(router_client, model="router-model"),
        renderer=PromptRenderer(),
        verification_agent=VerificationAgent(
            client=verification_client,
            model="verification-model",
            prompt_renderer=PromptRenderer(),
            lookup_customer_handler=lookup_customer_handler,
        ),
    )

    with pytest.raises(SpecialistNotAvailableError, match="policy"):
        await service(context=make_context(), user_message="What are your hours?")


@pytest.mark.asyncio
async def test_router_failure_does_not_invoke_verification_agent() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    router_client = FakeOpenAIClient(FakeOpenAIResponses(request_error=RuntimeError("private router detail")))
    verification_client = FakeOpenAIClient(FakeOpenAIResponses(responses=[]))
    lookup_customer_handler = create_lookup_customer_handler(
        FakeLegacyCRMClient(),
        conversation_state=state,
        conversation_store=store,
    )
    service = route_and_respond(
        router=OpenAIRouter(router_client, model="router-model"),
        renderer=PromptRenderer(),
        verification_agent=VerificationAgent(
            client=verification_client,
            model="verification-model",
            prompt_renderer=PromptRenderer(),
            lookup_customer_handler=lookup_customer_handler,
        ),
    )

    with pytest.raises(RouterRequestError, match="Router request failed") as error:
        await service(context=make_context(), user_message="My phone number is 321-555-0100.")

    assert "private router detail" not in str(error.value)
    assert verification_client.responses.create_calls == []


@pytest.mark.asyncio
async def test_verification_failure_is_not_reported_as_router_failure() -> None:
    store = InMemoryConversationStore()
    state = store.create("conv-test")
    store.save = Mock(wraps=store.save)
    router_client = FakeOpenAIClient(
        FakeOpenAIResponses(parsed=RouteDecision(destination="verification", reason="Caller supplied a phone number."))
    )
    verification_client = FakeOpenAIClient(
        FakeOpenAIResponses(
            request_error=RuntimeError("private verification detail"),
        )
    )
    lookup_customer_handler = create_lookup_customer_handler(
        FakeLegacyCRMClient(),
        conversation_state=state,
        conversation_store=store,
    )
    service = route_and_respond(
        router=OpenAIRouter(router_client, model="router-model"),
        renderer=PromptRenderer(),
        verification_agent=VerificationAgent(
            client=verification_client,
            model="verification-model",
            prompt_renderer=PromptRenderer(),
            lookup_customer_handler=lookup_customer_handler,
        ),
    )

    with pytest.raises(AgentModelRequestError, match="Responses API request failed"):
        await service(context=make_context(), user_message="My phone number is 321-555-0100.")

    assert len(verification_client.responses.create_calls) == 1
