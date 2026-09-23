from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app.dependencies import get_legacy_crm_client, get_scheduler_client
from app.integrations import (
    CallerConfirmationRequiredError,
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

