"""Shared runtime credentials with a mocked get_openai_api_key; no remote services."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app import dependencies
from app.agents.prompt_context import PromptContext
from app.main import lifespan


@pytest.fixture
def configured(monkeypatch):
    client = SimpleNamespace(close=AsyncMock())
    key_loader = Mock(return_value="dummy-runtime-secret")
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(dependencies, "get_openai_api_key", key_loader)
    monkeypatch.setattr(dependencies, "AsyncOpenAI", client_factory)
    monkeypatch.setenv("PAWLINE_ROUTER_BACKEND", "responses")
    request = Request({"type": "http", "app": FastAPI(lifespan=lifespan)})
    return SimpleNamespace(
        request=request, client=client, key_loader=key_loader, client_factory=client_factory,
    )


@pytest.mark.asyncio
async def test_concurrent_requests_retrieve_once_off_event_loop_and_reuse_client(configured):
    setup = configured
    loop_thread = threading.get_ident()
    worker_threads = []

    def get_key():
        worker_threads.append(threading.get_ident())
        return "dummy-runtime-secret"

    setup.key_loader.side_effect = get_key
    clients = await asyncio.gather(*(
        dependencies.get_openai_client(setup.request) for _ in range(5)
    ))
    assert all(client is setup.client for client in clients)
    assert worker_threads and all(thread != loop_thread for thread in worker_threads)
    setup.key_loader.assert_called_once_with()
    setup.client_factory.assert_called_once_with(api_key="dummy-runtime-secret")


@pytest.mark.asyncio
async def test_runtime_requests_and_turns_reuse_client_without_exposing_secret(configured, monkeypatch, caplog, capsys):
    setup = configured
    monkeypatch.setattr(dependencies, "get_legacy_crm_client", Mock())
    monkeypatch.setattr(dependencies, "get_scheduler_client", Mock())
    monkeypatch.setattr(dependencies, "get_handoff_client", Mock())
    policy = SimpleNamespace(respond=AsyncMock(return_value="Clinic hours are available."))
    monkeypatch.setattr(dependencies, "get_policy_agent", AsyncMock(return_value=policy))
    router = SimpleNamespace(route=AsyncMock(return_value=SimpleNamespace(destination="policy")))
    monkeypatch.setattr(dependencies, "create_router", Mock(return_value=router))
    runtime = await dependencies.get_prompt_agent_runtime(setup.request)
    assert runtime._openai_client is setup.client
    next_request = Request({"type": "http", "app": setup.request.app})
    assert await dependencies.get_prompt_agent_runtime(next_request) is runtime
    events = []
    runtime._workflow_observer = events.append
    await runtime.start_conversation("demo-conversation")
    await runtime.handle_turn("demo-conversation", "What are the clinic hours?")
    await runtime.handle_turn("demo-conversation", "Are those weekday hours?")
    setup.key_loader.assert_called_once_with()
    setup.client_factory.assert_called_once_with(api_key="dummy-runtime-secret")
    assert policy.respond.await_count == 2
    context = policy.respond.call_args.kwargs["context"]
    assert isinstance(context, PromptContext)
    assert "dummy-runtime-secret" not in context.model_dump_json() + repr(events) + caplog.text
    assert capsys.readouterr() == ("", "")


@pytest.mark.asyncio
async def test_policy_dependencies_share_client_and_retriever(configured, monkeypatch):
    setup = configured
    retriever = SimpleNamespace(index=AsyncMock())
    embedding_factory = Mock()
    generator_factory = Mock()
    monkeypatch.setattr(dependencies, "OpenAIEmbeddingProvider", embedding_factory)
    monkeypatch.setattr(dependencies, "OpenAIPolicyAnswerGenerator", generator_factory)
    monkeypatch.setattr(dependencies, "PolicyRetriever", Mock(return_value=retriever))
    monkeypatch.setattr(dependencies, "load_policy_chunks", Mock(return_value=[]))
    agent, service = await asyncio.gather(
        dependencies.get_policy_agent(setup.request),
        dependencies.get_policy_answer_service(setup.request),
    )
    assert agent._client is setup.client
    assert service._retriever is retriever
    assert embedding_factory.call_args.kwargs["client"] is setup.client
    assert generator_factory.call_args.kwargs["client"] is setup.client
    retriever.index.assert_awaited_once_with([])
    setup.key_loader.assert_called_once_with()
    setup.client_factory.assert_called_once_with(api_key="dummy-runtime-secret")


def test_lazy_initialization_and_shutdown_closes_shared_client(configured):
    setup = configured
    app = setup.request.app

    @app.get("/initialize")
    async def initialize(request: Request):
        await dependencies.get_openai_client(request)
        return {"status": "ok"}

    with TestClient(app) as client:
        setup.key_loader.assert_not_called()
        for _ in range(2):
            assert client.get("/initialize").json() == {"status": "ok"}
        setup.key_loader.assert_called_once_with()
        setup.client.close.assert_not_awaited()
    setup.client.close.assert_awaited_once_with()
    assert app.state.openai_client is None
    assert app.state.prompt_agent_runtime is None


@pytest.mark.parametrize("dependency", [
    "get_prompt_agent_runtime", "get_policy_agent", "get_policy_answer_service", "get_policy_retriever",
])
@pytest.mark.asyncio
async def test_helper_failure_keeps_services_unavailable_without_fallback_or_leaks(configured, dependency, monkeypatch, caplog, capsys):
    setup = configured
    monkeypatch.setenv("OPENAI_API_KEY", "unused-environment-secret")
    setup.key_loader.side_effect = RuntimeError("PRIVATE_CREDENTIAL PRIVATE_RESPONSE")
    assert await getattr(dependencies, dependency)(setup.request) is None
    setup.client_factory.assert_not_called()
    assert caplog.text == ""
    assert capsys.readouterr() == ("", "")


@pytest.mark.asyncio
async def test_client_initialization_failure_remains_safe(configured, caplog, capsys):
    setup = configured
    setup.client_factory.side_effect = ValueError("PRIVATE_CREDENTIAL PRIVATE_RESPONSE")
    assert await dependencies.get_openai_client(setup.request) is None
    assert caplog.text == ""
    assert capsys.readouterr() == ("", "")


def test_health_and_unavailable_agent_response_when_helper_fails(monkeypatch):
    from app.main import app

    key_loader = Mock(side_effect=RuntimeError("PRIVATE_CREDENTIAL PRIVATE_RESPONSE"))
    monkeypatch.setattr(dependencies, "get_openai_api_key", key_loader)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        key_loader.assert_not_called()
        response = client.post("/conversations")
        assert response.status_code == 503
        assert response.json() == {"detail": "agent-service-unavailable"}


@pytest.fixture
def real_initialization(monkeypatch):
    """Exercise real dependency constructors and policy indexing, faking only credentials/API."""
    from app.main import app

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAWLINE_ROUTER_BACKEND", "responses")
    for name in ("LEGACY_CRM_API_KEY", "SCHEDULER_JWT_SECRET", "HANDOFF_WEBHOOK_SECRET"):
        monkeypatch.setenv(name, "dummy-integration-credential-at-least-32-bytes")

    async def embed(**kwargs):
        return SimpleNamespace(data=[
            SimpleNamespace(index=index, embedding=[1.0, 0.5])
            for index in range(len(kwargs["input"]))
        ])

    client = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock(side_effect=embed)), close=AsyncMock())
    key_loader = Mock(return_value="dummy-infISical-runtime-key")
    factory = Mock(return_value=client)
    monkeypatch.setattr(dependencies, "get_openai_api_key", key_loader)
    monkeypatch.setattr(dependencies, "AsyncOpenAI", factory)
    for name in (
        "openai_client", "openai_client_lock", "prompt_agent_runtime", "prompt_agent_runtime_lock",
        "policy_agent", "policy_agent_lock", "policy_retriever", "policy_retriever_lock",
        "policy_answer_service", "policy_answer_service_lock", "conversation_store",
    ):
        monkeypatch.setattr(app.state, name, None, raising=False)
    return SimpleNamespace(app=app, client=client, key_loader=key_loader, factory=factory)


def test_post_conversations_initializes_real_policy_chain_without_environment_openai_key(real_initialization):
    setup = real_initialization
    with TestClient(setup.app) as client:
        assert client.post("/conversations").status_code == 201
        runtime = setup.app.state.prompt_agent_runtime
        retriever = setup.app.state.policy_retriever
        assert retriever.indexed_chunks
        assert runtime._openai_client is runtime._policy_agent._client is setup.client
        assert runtime._router._client is setup.client
        assert retriever._embedding_provider._client is setup.client
        assert client.post("/conversations").status_code == 201
        assert setup.app.state.prompt_agent_runtime is runtime
        setup.key_loader.assert_called_once_with()
        setup.factory.assert_called_once_with(api_key="dummy-infISical-runtime-key")
        setup.client.embeddings.create.assert_awaited_once()
        request = setup.client.embeddings.create.call_args.kwargs
        assert request["model"] == "text-embedding-3-small"
        assert request["encoding_format"] == "float"
        assert len(request["input"]) == len(retriever.indexed_chunks)
        setup.client.close.assert_not_awaited()
    setup.client.close.assert_awaited_once_with()


@pytest.mark.parametrize("missing", ["LEGACY_CRM_API_KEY", "SCHEDULER_JWT_SECRET", "HANDOFF_WEBHOOK_SECRET"])
def test_missing_integration_configuration_reproduces_503_and_closes_client(real_initialization, monkeypatch, missing, caplog, capsys):
    setup = real_initialization
    monkeypatch.delenv(missing)
    with TestClient(setup.app) as client:
        response = client.post("/conversations")
        assert response.status_code == 503
        assert response.json() == {"detail": "agent-service-unavailable"}
        setup.key_loader.assert_called_once_with()
        setup.client.embeddings.create.assert_not_awaited()
        setup.client.close.assert_awaited_once_with()
        assert setup.app.state.openai_client is None
        assert setup.app.state.prompt_agent_runtime is None
        assert setup.app.state.prompt_agent_runtime_lock is not None
    setup.client.close.assert_awaited_once_with()
    assert "dummy-infISical-runtime-key" not in caplog.text
    assert capsys.readouterr() == ("", "")


def test_embedding_failure_keeps_sanitized_503_and_closes_client(real_initialization, caplog, capsys):
    setup = real_initialization
    setup.client.embeddings.create.side_effect = RuntimeError("PRIVATE_CREDENTIAL PRIVATE_POLICY PRIVATE_BODY")
    with TestClient(setup.app) as client:
        response = client.post("/conversations")
        assert response.status_code == 503
        assert response.json() == {"detail": "agent-service-unavailable"}
        setup.client.close.assert_awaited_once_with()
        assert setup.app.state.policy_agent is None
        assert setup.app.state.policy_retriever is None
        assert setup.app.state.openai_client is None
    assert "PRIVATE_" not in caplog.text
    assert capsys.readouterr() == ("", "")


@pytest.mark.asyncio
async def test_failed_initialization_retains_waiting_runtime_lock(configured, monkeypatch):
    setup = configured
    entered = asyncio.Event()
    release = asyncio.Event()

    async def unavailable_policy(request):
        entered.set()
        await release.wait()

    monkeypatch.setattr(dependencies, "get_legacy_crm_client", Mock())
    monkeypatch.setattr(dependencies, "get_scheduler_client", Mock())
    monkeypatch.setattr(dependencies, "get_handoff_client", Mock())
    monkeypatch.setattr(dependencies, "get_policy_agent", unavailable_policy)
    first = asyncio.create_task(dependencies.get_prompt_agent_runtime(setup.request))
    await asyncio.wait_for(entered.wait(), timeout=1)
    lock = setup.request.app.state.prompt_agent_runtime_lock
    second = asyncio.create_task(dependencies.get_prompt_agent_runtime(setup.request))
    release.set()
    assert await asyncio.gather(first, second) == [None, None]
    assert setup.request.app.state.prompt_agent_runtime_lock is lock
    assert setup.client.close.await_count == 2
