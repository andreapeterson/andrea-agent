import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app.agents.prompt_renderer import PromptRenderingError
from app.agents.responses_tool_loop import (
    AgentModelRequestError,
    MalformedToolCallError,
    ToolExecutionError,
    ToolLoopLimitError,
    UnknownAgentToolError,
)
from app.agents.router_agent import RouterRequestError, RouterResponseError
from app.agents.router_handoff import SpecialistNotAvailableError
from app.dependencies import (
    get_handoff_client,
    get_legacy_crm_client,
    get_policy_answer_service,
    get_prompt_agent_runtime,
    get_scheduler_client,
)
from app.integrations import (
    CallerConfirmationRequiredError,
    HandoffClient,
    HandoffNotRequiredError,
    HandoffRequestError,
    HandoffResponseError,
    IdempotencyConflictError,
    LegacyCRMParseError,
    LegacyCRMRequestError,
    PolicyGenerationRequestError,
    PolicyGenerationResponseError,
    SchedulerClient,
    SchedulerRequestError,
    SchedulerResponseError,
    SlotNotFoundError,
    SlotUnavailableError,
)
from app.models.appointment import AppointmentSlot, AppointmentType, BookingConfirmation, BookingRequest
from app.models.conversation_api import ConversationResponse, ConversationTurnRequest, conversation_response_from_result
from app.models.customer import Customer
from app.models.handoff import HandoffCreateRequest, HandoffReceipt, HandoffRequest
from app.models.policy_answer import PolicyAnswerResponse, PolicyQuestionRequest
from app.models.routing import RoutingAction
from app.services import (
    ConversationNotFoundError,
    PolicyAnswerService,
)
from app.services.conversation_store import ConversationAlreadyExistsError
from app.services.handoff_summary import build_handoff_summary
from app.services.policy_retriever import PolicyRetrievalError
from app.services.prompt_agent_runtime import PromptAgentRuntime
from app.services.routing import assess_routing

app = FastAPI(title="PawLine")

PromptAgentRuntimeDep = Annotated[PromptAgentRuntime | None, Depends(get_prompt_agent_runtime)]


@app.post("/conversations", response_model=ConversationResponse, status_code=201)
async def start_conversation_route(
    runtime: PromptAgentRuntimeDep,
) -> ConversationResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent-service-unavailable")

    try:
        result = await runtime.start_conversation("conversation-" + uuid.uuid4().hex)
    except ConversationAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail="invalid-conversation-state") from exc

    return conversation_response_from_result(result)


@app.post("/conversations/{conversation_id}/messages", response_model=ConversationResponse)
async def send_conversation_message(
    conversation_id: str,
    turn_request: ConversationTurnRequest,
    runtime: PromptAgentRuntimeDep,
) -> ConversationResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="agent-service-unavailable")

    try:
        result = await runtime.handle_turn(conversation_id, turn_request.message)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation-not-found") from exc
    except (RouterRequestError, AgentModelRequestError) as exc:
        raise HTTPException(status_code=503, detail="agent-service-unavailable") from exc
    except (RouterResponseError, MalformedToolCallError, ToolLoopLimitError, ValueError) as exc:
        # The existing specialist loop uses ValueError for missing or malformed model output.
        raise HTTPException(status_code=502, detail="invalid-agent-response") from exc
    except (UnknownAgentToolError, ToolExecutionError, SpecialistNotAvailableError) as exc:
        raise HTTPException(status_code=503, detail="agent-tool-unavailable") from exc
    except PromptRenderingError as exc:
        raise HTTPException(status_code=500, detail="agent-configuration-error") from exc

    return conversation_response_from_result(result)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/customers/lookup", response_model=Customer)
async def lookup_customer(
    phone: str = Query(..., description="Customer phone number."),
    crm_client=Depends(get_legacy_crm_client),
) -> Customer:
    try:
        customer = await crm_client.find_customer_by_phone(phone)
    except LegacyCRMRequestError as error:
        raise HTTPException(status_code=503, detail="customer-service-unavailable") from error
    except LegacyCRMParseError as error:
        raise HTTPException(status_code=502, detail="invalid-upstream-data") from error

    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    return customer


@app.get("/appointments/slots", response_model=list[AppointmentSlot])
async def list_appointment_slots(
    pet_id: str = Query(..., description="Pet identifier."),
    appointment_type: AppointmentType = Query(..., description="Appointment type."),
    scheduler_client: SchedulerClient = Depends(get_scheduler_client),
) -> list[AppointmentSlot]:
    try:
        return await scheduler_client.find_slots(pet_id, appointment_type)
    except SchedulerResponseError as error:
        raise HTTPException(status_code=502, detail="invalid-scheduler-data") from error
    except SchedulerRequestError as error:
        raise HTTPException(status_code=503, detail="scheduling-service-unavailable") from error


@app.post("/appointments/bookings", response_model=BookingConfirmation, status_code=201)
async def create_appointment_booking(
    booking_request: BookingRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
    scheduler_client: SchedulerClient = Depends(get_scheduler_client),
) -> BookingConfirmation:
    try:
        return await scheduler_client.book_appointment(booking_request, idempotency_key)
    except SlotNotFoundError as error:
        raise HTTPException(status_code=404, detail="Slot not found") from error
    except CallerConfirmationRequiredError as error:
        raise HTTPException(status_code=400, detail="caller-confirmation-required") from error
    except SlotUnavailableError as error:
        raise HTTPException(status_code=409, detail="slot-unavailable") from error
    except IdempotencyConflictError as error:
        raise HTTPException(status_code=409, detail="idempotency-key-reused") from error
    except SchedulerResponseError as error:
        raise HTTPException(status_code=502, detail="invalid-scheduler-data") from error
    except SchedulerRequestError as error:
        raise HTTPException(status_code=503, detail="scheduling-service-unavailable") from error


@app.post("/handoffs", response_model=HandoffReceipt, status_code=202)
async def create_handoff(
    handoff_create_request: HandoffCreateRequest,
    handoff_client: HandoffClient = Depends(get_handoff_client),
) -> HandoffReceipt:
    routing_decision = assess_routing(handoff_create_request.intake_answers)
    if routing_decision.next_action != RoutingAction.CREATE_HANDOFF:
        raise HTTPException(status_code=409, detail="handoff-not-required")

    summary = build_handoff_summary(
        handoff_create_request.original_concern,
        handoff_create_request.intake_answers,
        routing_decision,
    )

    handoff_request = HandoffRequest(
        conversation_id=handoff_create_request.conversation_id,
        verified_customer_id=handoff_create_request.verified_customer_id,
        selected_pet_id=handoff_create_request.selected_pet_id,
        original_concern=handoff_create_request.original_concern,
        intake_answers=handoff_create_request.intake_answers,
        routing_decision=routing_decision,
        summary=summary,
    )

    try:
        return await handoff_client.create_handoff(handoff_request)
    except HandoffNotRequiredError as error:
        raise HTTPException(status_code=409, detail="handoff-not-required") from error
    except HandoffResponseError as error:
        raise HTTPException(status_code=502, detail="invalid-handoff-data") from error
    except HandoffRequestError as error:
        raise HTTPException(status_code=503, detail="handoff-service-unavailable") from error


@app.post("/policies/answer", response_model=PolicyAnswerResponse)
async def answer_policy_question(
    question: PolicyQuestionRequest,
    policy_answer_service: PolicyAnswerService | None = Depends(get_policy_answer_service),
) -> PolicyAnswerResponse:
    if policy_answer_service is None:
        raise HTTPException(status_code=503, detail="policy-service-unavailable")

    try:
        return await policy_answer_service.answer(question.question)
    except (RuntimeError, PolicyRetrievalError, PolicyGenerationRequestError) as exc:
        raise HTTPException(status_code=503, detail="policy-service-unavailable") from exc
    except PolicyGenerationResponseError as exc:
        raise HTTPException(status_code=502, detail="invalid-policy-response") from exc

