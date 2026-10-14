"""Handoff domain models for PawLine's fictional human escalation workflow."""

from datetime import datetime
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .routing import IntakeAnswers, RoutingDecision, RoutingLevel


class HandoffSummary(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    original_concern: str = Field(..., min_length=1)
    routing_level: RoutingLevel
    positive_signals: list[str] = Field(default_factory=list)
    negative_signals: list[str] = Field(default_factory=list)
    unanswered_signals: list[str] = Field(default_factory=list)
    matched_rule_ids: list[str] = Field(default_factory=list)


class HandoffCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    conversation_id: str = Field(..., min_length=1)
    verified_customer_id: str = Field(..., min_length=1)
    selected_pet_id: str = Field(..., min_length=1)
    original_concern: str = Field(..., min_length=1)
    intake_answers: IntakeAnswers


class HandoffReason(str, Enum):
    URGENT_INTAKE = "urgent_intake"
    CALLER_REQUEST = "caller_request"
    OUT_OF_SCOPE = "out_of_scope"
    SAFETY_CONCERN = "safety_concern"


class HandoffRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    conversation_id: str = Field(..., min_length=1)
    reason: HandoffReason = HandoffReason.URGENT_INTAKE
    explanation: str | None = Field(default=None, min_length=1)
    verified_customer_id: str | None = Field(default=None, min_length=1)
    selected_pet_id: str | None = Field(default=None, min_length=1)
    original_concern: str | None = Field(default=None, min_length=1)
    intake_answers: IntakeAnswers | None = None
    routing_decision: RoutingDecision | None = None
    summary: HandoffSummary | None = None

    @model_validator(mode="after")
    def require_reason_details(self) -> Self:
        if self.reason == HandoffReason.URGENT_INTAKE:
            if self.intake_answers is None:
                raise ValueError("Urgent intake handoffs require recorded intake answers.")
        elif self.explanation is None:
            raise ValueError("General handoffs require an explanation.")
        return self


class HandoffStatus(str, Enum):
    QUEUED = "queued"


class HandoffReceipt(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    case_id: str = Field(..., min_length=1)
    conversation_id: str = Field(..., min_length=1)
    status: HandoffStatus
    received_at: datetime
