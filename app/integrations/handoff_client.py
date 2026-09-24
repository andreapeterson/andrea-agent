import hashlib
import hmac
import time
from datetime import datetime

import httpx
from pydantic import ValidationError

from app.models.handoff import HandoffReceipt, HandoffRequest


class HandoffClientError(RuntimeError):
    """Base exception for handoff client failures."""


class HandoffRequestError(HandoffClientError):
    """Raised when the handoff service rejects or fails the request."""


class HandoffResponseError(HandoffClientError):
    """Raised when the handoff service returns malformed or invalid receipt data."""


class HandoffNotRequiredError(HandoffRequestError):
    """Raised when the upstream service refuses to hand off a non-urgent request."""


class HandoffClient:
    def __init__(
        self,
        base_url: str,
        webhook_secret: str,
        timeout_seconds: float = 5.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._webhook_secret = webhook_secret
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def create_handoff(
        self,
        handoff_request: HandoffRequest,
    ) -> HandoffReceipt:
        body_bytes = handoff_request.model_dump_json().encode("utf-8")
        timestamp = str(int(time.time()))
        signed_message = timestamp.encode("utf-8") + b"." + body_bytes
        digest = hmac.new(
            self._webhook_secret.encode("utf-8"),
            signed_message,
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "X-PawLine-Timestamp": timestamp,
            "X-PawLine-Signature": f"sha256={digest}",
        }

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                response = await client.post(
                    "/handoffs",
                    content=body_bytes,
                    headers=headers,
                )
            except httpx.RequestError as exc:
                raise HandoffRequestError("Handoff service request failed.") from exc

        if response.status_code == 202:
            try:
                payload = response.json()
            except ValueError as exc:
                raise HandoffResponseError("Handoff service returned invalid JSON.") from exc
            try:
                return HandoffReceipt.model_validate(payload)
            except ValidationError as exc:
                raise HandoffResponseError("Handoff service response did not match the expected receipt schema.") from exc

        if response.status_code == 409:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            if detail == "handoff-not-required":
                raise HandoffNotRequiredError("Handoff is not required.")

        if response.status_code == 401:
            raise HandoffRequestError("Handoff service rejected the request signature.")

        if response.status_code >= 500:
            raise HandoffRequestError(f"Handoff service returned status {response.status_code}.")

        raise HandoffRequestError(f"Handoff service returned unexpected status {response.status_code}.")
