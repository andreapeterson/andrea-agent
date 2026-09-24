"""Handoff domain models for PawLine's fictional human escalation workflow."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

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


class HandoffRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    conversation_id: str = Field(..., min_length=1)
    verified_customer_id: str = Field(..., min_length=1)
    selected_pet_id: str = Field(..., min_length=1)
    original_concern: str = Field(..., min_length=1)
    intake_answers: IntakeAnswers
    routing_decision: RoutingDecision
    summary: HandoffSummary


class HandoffStatus(str, Enum):
    QUEUED = "queued"


class HandoffReceipt(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    case_id: str = Field(..., min_length=1)
    conversation_id: str = Field(..., min_length=1)
    status: HandoffStatus
    received_at: datetime
