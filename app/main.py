from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app.dependencies import get_handoff_client, get_legacy_crm_client, get_scheduler_client
from app.integrations import (
    CallerConfirmationRequiredError,
    HandoffClient,
    HandoffNotRequiredError,
    HandoffRequestError,
    HandoffResponseError,
    IdempotencyConflictError,
    LegacyCRMParseError,
    LegacyCRMRequestError,
    SchedulerClient,
    SchedulerRequestError,
    SchedulerResponseError,
    SlotNotFoundError,
    SlotUnavailableError,
)
from app.models.appointment import AppointmentSlot, AppointmentType, BookingConfirmation, BookingRequest
from app.models.customer import Customer
from app.models.handoff import HandoffCreateRequest, HandoffReceipt, HandoffRequest
from app.models.routing import RoutingAction
from app.services.routing import assess_routing

app = FastAPI(title="PawLine")


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

    handoff_request = HandoffRequest(
        conversation_id=handoff_create_request.conversation_id,
        verified_customer_id=handoff_create_request.verified_customer_id,
        selected_pet_id=handoff_create_request.selected_pet_id,
        original_concern=handoff_create_request.original_concern,
        intake_answers=handoff_create_request.intake_answers,
        routing_decision=routing_decision,
    )

    try:
        return await handoff_client.create_handoff(handoff_request)
    except HandoffNotRequiredError as error:
        raise HTTPException(status_code=409, detail="handoff-not-required") from error
    except HandoffResponseError as error:
        raise HTTPException(status_code=502, detail="invalid-handoff-data") from error
    except HandoffRequestError as error:
        raise HTTPException(status_code=503, detail="handoff-service-unavailable") from error

