from datetime import datetime, timedelta, timezone

import httpx
import jwt
from pydantic import ValidationError

from app.models.appointment import AppointmentSlot, AppointmentType, BookingConfirmation, BookingRequest


class SchedulerError(RuntimeError):
    """Base error for scheduler API failures."""


class SchedulerRequestError(SchedulerError):
    """Raised when the scheduler request fails or returns an unexpected HTTP status."""


class SchedulerResponseError(SchedulerError):
    """Raised when the scheduler returns invalid JSON or schema-mismatched data."""


class SchedulingConflictError(SchedulerRequestError):
    """Raised when the scheduler rejects a booking because of a conflict."""


class SlotUnavailableError(SchedulingConflictError):
    """Raised when the requested slot is no longer available."""


class IdempotencyConflictError(SchedulingConflictError):
    """Raised when the idempotency key is reused with different data."""


class SlotNotFoundError(SchedulerRequestError):
    """Raised when the scheduler cannot find the requested slot."""


class CallerConfirmationRequiredError(SchedulerRequestError):
    """Raised when the caller must confirm booking before the scheduler accepts it."""


class SchedulerClient:
    def __init__(
        self,
        base_url: str,
        jwt_secret: str,
        timeout_seconds: float = 5.0,
        *, #every parameter after this must be passed as a keyword argument i.e. transport=None
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._jwt_secret = jwt_secret
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    def _create_jwt(self) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "pawline-service", #subject of the token, identifying the service making the request
            "iss": "pawline", #issuer of the token, identifying the entity that issued the token
            "aud": "mock-scheduler", #audience of the token, identifying the intended recipient of the token
            "iat": int(now.timestamp()), #issued at time, indicating when the token was issued
            "exp": int((now + timedelta(minutes=5)).timestamp()), 
        }
        return jwt.encode(payload, self._jwt_secret, algorithm="HS256") #sign the token using the HS256 algorithm and the provided secret

    def _authorization_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._create_jwt()}"}

    async def find_slots(
        self,
        pet_id: str,
        appointment_type: AppointmentType,
    ) -> list[AppointmentSlot]:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                response = await client.get(
                    "/slots",
                    params={"pet_id": pet_id, "appointment_type": appointment_type.value},
                    headers=self._authorization_headers(),
                )
            except httpx.RequestError as exc:
                raise SchedulerRequestError("Scheduler request failed.") from exc

        if response.status_code != 200:
            raise SchedulerRequestError(f"Scheduler request failed with status {response.status_code}.")

        try:
            payload = response.json()
        except ValueError as exc:
            raise SchedulerResponseError("Scheduler response contained invalid JSON.") from exc

        if not isinstance(payload, list):
            raise SchedulerResponseError("Scheduler response did not contain a list of slots.")

        try:
            return [AppointmentSlot.model_validate(item) for item in payload]
        except ValidationError as exc:
            raise SchedulerResponseError("Scheduler response did not match the expected slot schema.") from exc

    async def book_appointment(
        self,
        booking_request: BookingRequest,
        idempotency_key: str,
    ) -> BookingConfirmation:
        headers = self._authorization_headers()
        headers["Idempotency-Key"] = idempotency_key

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                response = await client.post(
                    "/bookings",
                    json=booking_request.model_dump(mode="json"),
                    headers=headers,
                )
            except httpx.RequestError as exc:
                raise SchedulerRequestError("Scheduler request failed.") from exc

        if response.status_code == 201:
            try:
                payload = response.json()
            except ValueError as exc:
                raise SchedulerResponseError("Scheduler response contained invalid JSON.") from exc
            try:
                return BookingConfirmation.model_validate(payload)
            except ValidationError as exc:
                raise SchedulerResponseError("Scheduler response did not match the expected booking schema.") from exc

        if response.status_code == 404:
            raise SlotNotFoundError("Slot not found.")

        if response.status_code == 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            if detail == "caller-confirmation-required":
                raise CallerConfirmationRequiredError("Caller confirmation required.")

        if response.status_code == 409:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            if detail == "slot-unavailable":
                raise SlotUnavailableError("Slot unavailable.")
            if detail == "idempotency-key-reused":
                raise IdempotencyConflictError("Idempotency key reused.")

        raise SchedulerRequestError(f"Scheduler request failed with status {response.status_code}.")
