import hashlib
import hmac
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status

from app.models.handoff import HandoffReceipt, HandoffRequest, HandoffStatus
from app.models.routing import RoutingAction, RoutingLevel

app = FastAPI(title="Mock Human Handoff API")

MAX_CLOCK_SKEW_SECONDS = 300
MAX_FUTURE_SECONDS = 60


def _handoff_secret() -> str:
    return os.getenv("MOCK_HANDOFF_WEBHOOK_SECRET", "dev-handoff-secret")


def _invalid_signature() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing signature")


async def verify_handoff_signature(request: Request) -> None:
    timestamp = request.headers.get("X-PawLine-Timestamp")
    signature = request.headers.get("X-PawLine-Signature")

    if timestamp is None or signature is None:
        raise _invalid_signature()

    try:
        ts_value = int(timestamp)
    except ValueError as exc:
        raise _invalid_signature() from exc

    now_seconds = int(time.time())
    if abs(now_seconds - ts_value) > MAX_CLOCK_SKEW_SECONDS:
        raise _invalid_signature()
    if ts_value > now_seconds + MAX_FUTURE_SECONDS:
        raise _invalid_signature()

    raw_body = await request.body()
    signed_message = timestamp.encode("utf-8") + b"." + raw_body
    expected_digest = hmac.new(
        _handoff_secret().encode("utf-8"),
        signed_message,
        hashlib.sha256,
    ).hexdigest()
    expected_signature = f"sha256={expected_digest}"

    if not hmac.compare_digest(signature, expected_signature):
        raise _invalid_signature()


@app.post("/handoffs", response_model=HandoffReceipt, status_code=status.HTTP_202_ACCEPTED)
async def create_handoff(
    request: HandoffRequest,
    _verified: None = Depends(verify_handoff_signature),
) -> HandoffReceipt:
    if request.routing_decision.routing_level != RoutingLevel.URGENT or request.routing_decision.next_action != RoutingAction.CREATE_HANDOFF:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="handoff-not-required")

    case_id = f"case_{uuid4().hex}"
    return HandoffReceipt(
        case_id=case_id,
        conversation_id=request.conversation_id,
        status=HandoffStatus.QUEUED,
        received_at=datetime.now(timezone.utc),
    )
