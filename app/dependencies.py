import asyncio
import os
import uuid
from pathlib import Path

from fastapi import Request
from openai import AsyncOpenAI

from app.agents.policy_agent import PolicyAgent
from app.agents.prompt_renderer import PromptRenderer
from app.agents.search_policy_tool import create_search_policy_handler
from app.integrations import (
    HandoffClient,
    LegacyCRMClient,
    OpenAIEmbeddingProvider,
    OpenAIPolicyAnswerGenerator,
    SchedulerClient,
)
from app.integrations.turn_interpreter import OpenAITurnInterpreter
from app.services import AgentOrchestrator, InMemoryConversationStore, PolicyAnswerService
from app.services.policy_loader import load_policy_chunks
from app.services.policy_retriever import PolicyRetriever


async def get_agent_orchestrator(request: Request) -> AgentOrchestrator | None:
    app_state = request.app.state
    orchestrator = getattr(app_state, "agent_orchestrator", None)
    if orchestrator is not None:
        return orchestrator

    lock = getattr(app_state, "agent_orchestrator_lock", None)
    if lock is None:
        app_state.agent_orchestrator_lock = asyncio.Lock()
        lock = app_state.agent_orchestrator_lock

    async with lock:
        orchestrator = getattr(app_state, "agent_orchestrator", None)
        if orchestrator is not None:
            return orchestrator

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            app_state.agent_orchestrator = None
            return None

        try:
            crm_client = get_legacy_crm_client()
            scheduler_client = get_scheduler_client()
            handoff_client = get_handoff_client()
            policy_service = await get_policy_answer_service(request)
            if policy_service is None:
                app_state.agent_orchestrator = None
                return None

            orchestrator = AgentOrchestrator(
                conversation_store=InMemoryConversationStore(),
                turn_interpreter=OpenAITurnInterpreter(
                    client=AsyncOpenAI(api_key=api_key),
                    model=os.getenv("OPENAI_TURN_MODEL", "gpt-6-luna"),
                ),
                legacy_crm_client=crm_client,
                scheduler_client=scheduler_client,
                handoff_client=handoff_client,
                policy_service=policy_service,
                idempotency_key_factory=lambda: uuid.uuid4().hex,
            )
        except Exception:
            app_state.agent_orchestrator = None
            return None

        app_state.agent_orchestrator = orchestrator
        return orchestrator


def get_legacy_crm_client() -> LegacyCRMClient:
    base_url = os.getenv("LEGACY_CRM_BASE_URL", "http://127.0.0.1:8001")
    api_key = os.getenv("LEGACY_CRM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LEGACY_CRM_API_KEY is not configured. Set it in the environment before starting the app."
        )
    return LegacyCRMClient(base_url=base_url, api_key=api_key)


def get_scheduler_client() -> SchedulerClient:
    base_url = os.getenv("SCHEDULER_BASE_URL", "http://127.0.0.1:8002")
    jwt_secret = os.getenv("SCHEDULER_JWT_SECRET")
    if not jwt_secret:
        raise RuntimeError(
            "SCHEDULER_JWT_SECRET is not configured. Set it in the environment before starting the app."
        )
    return SchedulerClient(base_url=base_url, jwt_secret=jwt_secret)


def get_handoff_client() -> HandoffClient:
    base_url = os.getenv("HANDOFF_BASE_URL", "http://127.0.0.1:8003")
    webhook_secret = os.getenv("HANDOFF_WEBHOOK_SECRET")
    if not webhook_secret:
        raise RuntimeError(
            "HANDOFF_WEBHOOK_SECRET is not configured. Set it in the environment before starting the app."
        )
    return HandoffClient(base_url=base_url, webhook_secret=webhook_secret)


async def get_policy_answer_service(request: Request) -> PolicyAnswerService | None:
    app_state = request.app.state
    service = getattr(app_state, "policy_answer_service", None)
    if service is not None:
        return service

    lock = getattr(app_state, "policy_answer_service_lock", None)
    if lock is None:
        app_state.policy_answer_service_lock = asyncio.Lock()
        lock = app_state.policy_answer_service_lock

    async with lock:
        service = getattr(app_state, "policy_answer_service", None)
        if service is not None:
            return service

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            app_state.policy_answer_service = None
            return None

        try:
            top_k = int(os.getenv("POLICY_TOP_K", "3"))
            minimum_similarity = float(os.getenv("POLICY_MIN_SIMILARITY", "0.45"))
        except ValueError:
            app_state.policy_answer_service = None
            return None

        try:
            client = AsyncOpenAI(api_key=api_key)
            policy_model = os.getenv("OPENAI_POLICY_MODEL", "gpt-6-luna")
            retriever = await get_policy_retriever(request)
            if retriever is None:
                return None
            generator = OpenAIPolicyAnswerGenerator(client=client, model=policy_model)
            service = PolicyAnswerService(
                retriever=retriever,
                generator=generator,
                top_k=top_k,
                minimum_similarity=minimum_similarity,
            )
        except Exception:  # pragma: no cover - real service initialization is tested through dependency override
            app_state.policy_answer_service = None
            return None

        app_state.policy_answer_service = service
        return service


async def get_policy_retriever(request: Request) -> PolicyRetriever | None:
    """Reuse one initialized policy index for HTTP answers and specialist searches."""
    app_state = request.app.state
    retriever = getattr(app_state, "policy_retriever", None)
    if retriever is not None:
        return retriever

    lock = getattr(app_state, "policy_retriever_lock", None)
    if lock is None:
        app_state.policy_retriever_lock = asyncio.Lock()
        lock = app_state.policy_retriever_lock

    async with lock:
        retriever = getattr(app_state, "policy_retriever", None)
        if retriever is not None:
            return retriever

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            return None

        try:
            client = AsyncOpenAI(api_key=api_key)
            embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            policy_directory = Path(__file__).resolve().parent.parent / "knowledge" / "clinic_policies"
            chunks = load_policy_chunks(policy_directory)
            embedding_provider = OpenAIEmbeddingProvider(client=client, model=embedding_model)
            retriever = PolicyRetriever(embedding_provider)
            await retriever.index(chunks)
        except Exception:
            return None

        app_state.policy_retriever = retriever
        return retriever


async def get_policy_agent(request: Request) -> PolicyAgent | None:
    """Wire a stateless policy specialist to the application's existing retriever."""
    app_state = request.app.state
    agent = getattr(app_state, "policy_agent", None)
    if agent is not None:
        return agent

    lock = getattr(app_state, "policy_agent_lock", None)
    if lock is None:
        app_state.policy_agent_lock = asyncio.Lock()
        lock = app_state.policy_agent_lock

    async with lock:
        agent = getattr(app_state, "policy_agent", None)
        if agent is not None:
            return agent

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            return None

        try:
            top_k = int(os.getenv("POLICY_TOP_K", "3"))
            minimum_similarity = float(os.getenv("POLICY_MIN_SIMILARITY", "0.45"))
            retriever = await get_policy_retriever(request)
            if retriever is None:
                return None

            search_policy_handler = create_search_policy_handler(
                retriever,
                top_k=top_k,
                minimum_similarity=minimum_similarity,
            )
            agent = PolicyAgent(
                client=AsyncOpenAI(api_key=api_key),
                model=os.getenv("OPENAI_POLICY_MODEL", "gpt-6-luna"),
                prompt_renderer=PromptRenderer(),
                search_policy_handler=search_policy_handler,
            )
        except Exception:
            return None

        app_state.policy_agent = agent
        return agent
