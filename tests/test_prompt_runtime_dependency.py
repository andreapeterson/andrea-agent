"""Check process-wide runtime initialization without contacting external services."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from fastapi import FastAPI, Request
from openai import AsyncOpenAI

from app import dependencies
from app.agents.policy_agent import PolicyAgent
from app.agents.prompt_renderer import PromptRenderer
from app.integrations.handoff_client import HandoffClient
from app.integrations.legacy_crm import LegacyCRMClient
from app.integrations.scheduler_client import SchedulerClient
from app.services.conversation_store import InMemoryConversationStore
from app.services.prompt_agent_runtime import PromptAgentRuntime


@pytest.fixture
def configured_dependencies(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
    monkeypatch.delenv("OPENAI_ROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_SPECIALIST_MODEL", raising=False)
    client = Mock(spec=AsyncOpenAI)
    crm = AsyncMock(spec=LegacyCRMClient)
    scheduler = AsyncMock(spec=SchedulerClient)
    handoff = AsyncMock(spec=HandoffClient)
    policy = Mock(spec=PolicyAgent)
    factories = {
        "AsyncOpenAI": Mock(return_value=client),
        "get_legacy_crm_client": Mock(return_value=crm),
        "get_scheduler_client": Mock(return_value=scheduler),
        "get_handoff_client": Mock(return_value=handoff),
        "get_policy_agent": AsyncMock(return_value=policy),
    }
    for name, factory in factories.items():
        monkeypatch.setattr(dependencies, name, factory)
    request = Request({"type": "http", "app": FastAPI()})
    return SimpleNamespace(
        request=request, client=client, crm=crm, scheduler=scheduler,
        handoff=handoff, policy=policy, factories=factories,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_models", [False, True])
async def test_runtime_and_store_are_cached_with_configured_clients_and_models(configured_dependencies, monkeypatch, custom_models) -> None:
    setup = configured_dependencies
    if custom_models:
        monkeypatch.setenv("OPENAI_ROUTER_MODEL", "custom-router")
        monkeypatch.setenv("OPENAI_SPECIALIST_MODEL", "custom-specialist")

    runtime = await dependencies.get_prompt_agent_runtime(setup.request)
    started = await runtime.start_conversation("shared-conversation")
    next_request = Request({"type": "http", "app": setup.request.app})
    reused = await dependencies.get_prompt_agent_runtime(next_request)

    assert isinstance(runtime, PromptAgentRuntime)
    assert runtime is reused is setup.request.app.state.prompt_agent_runtime
    assert isinstance(setup.request.app.state.conversation_store, InMemoryConversationStore)
    assert reused._conversation_store.get("shared-conversation") == started.state
    assert runtime._client is setup.client
    assert runtime._legacy_crm_client is setup.crm
    assert runtime._scheduler_client is setup.scheduler
    assert runtime._handoff_client is setup.handoff
    assert runtime._policy_agent is setup.policy
    assert isinstance(runtime._prompt_renderer, PromptRenderer)
    assert runtime._router._model == ("custom-router" if custom_models else "gpt-6-luna")
    assert runtime._specialist_model == ("custom-specialist" if custom_models else "gpt-6-luna")
    first_key = runtime._idempotency_key_factory()
    assert UUID(hex=first_key).hex == first_key
    assert runtime._idempotency_key_factory() != first_key
    setup.factories["AsyncOpenAI"].assert_called_once_with(api_key="fake-api-key")
    for name in ("get_legacy_crm_client", "get_scheduler_client", "get_handoff_client"):
        setup.factories[name].assert_called_once_with()
    setup.factories["get_policy_agent"].assert_awaited_once_with(setup.request)


@pytest.mark.asyncio
async def test_concurrent_initialization_builds_only_one_runtime(configured_dependencies) -> None:
    setup = configured_dependencies
    entered = asyncio.Event()
    release = asyncio.Event()

    async def prepare_policy(request):
        entered.set()
        await release.wait()
        return setup.policy

    setup.factories["get_policy_agent"].side_effect = prepare_policy
    first = asyncio.create_task(dependencies.get_prompt_agent_runtime(setup.request))
    await asyncio.wait_for(entered.wait(), timeout=1)
    second = asyncio.create_task(dependencies.get_prompt_agent_runtime(setup.request))
    try:
        await asyncio.sleep(0)
        setup.factories["get_policy_agent"].assert_awaited_once()
    finally:
        release.set()
        results = await asyncio.gather(first, second)
    assert results[0] is results[1] is setup.request.app.state.prompt_agent_runtime
    setup.factories["AsyncOpenAI"].assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [None, "", "  "])
async def test_missing_openai_configuration_returns_none_without_initializing_clients(configured_dependencies, monkeypatch, api_key) -> None:
    setup = configured_dependencies
    if api_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", api_key)

    assert await dependencies.get_prompt_agent_runtime(setup.request) is None
    for factory in setup.factories.values():
        factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_dependency", ["get_legacy_crm_client", "get_scheduler_client", "get_handoff_client", "get_policy_agent", "AsyncOpenAI"])
async def test_initialization_failure_returns_none_and_later_request_can_retry(configured_dependencies, failing_dependency) -> None:
    setup = configured_dependencies
    factory = setup.factories[failing_dependency]
    factory.side_effect = RuntimeError("private credential or upstream failure")

    assert await dependencies.get_prompt_agent_runtime(setup.request) is None
    assert getattr(setup.request.app.state, "prompt_agent_runtime", None) is None
    factory.side_effect = None
    assert isinstance(await dependencies.get_prompt_agent_runtime(setup.request), PromptAgentRuntime)


@pytest.mark.asyncio
async def test_unavailable_policy_agent_returns_none(configured_dependencies) -> None:
    setup = configured_dependencies
    setup.factories["get_policy_agent"].return_value = None
    assert await dependencies.get_prompt_agent_runtime(setup.request) is None
    setup.factories["AsyncOpenAI"].assert_not_called()


@pytest.mark.asyncio
async def test_existing_process_store_is_reused(configured_dependencies) -> None:
    setup = configured_dependencies
    store = InMemoryConversationStore()
    state = store.create("existing-conversation")
    setup.request.app.state.conversation_store = store

    runtime = await dependencies.get_prompt_agent_runtime(setup.request)

    assert runtime._conversation_store is store
    assert runtime._conversation_store.get("existing-conversation") == state
