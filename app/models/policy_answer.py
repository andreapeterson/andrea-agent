"""Public and internal models for grounded clinic-policy answers."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PolicyQuestionRequest(BaseModel):
    """A client question about clinic policy."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(..., min_length=1, description="Administrative clinic-policy question.")


class PolicyAnswerStatus(str, Enum):
    """Whether the service was able to answer using the available policy evidence."""

    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class PolicyCitation(BaseModel):
    """A verified citation that maps back to a retrieved chunk."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    document_title: str = Field(..., min_length=1)
    section_title: str = Field(..., min_length=1)
    source_name: str = Field(..., min_length=1)


class PolicyAnswerResponse(BaseModel):
    """Public, citation-safe answer for a clinic-policy question."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    status: PolicyAnswerStatus
    citations: list[PolicyCitation] = Field(default_factory=list)


class GeneratedPolicyAnswer(BaseModel):
    """Structured-output schema returned by the policy-answer model."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    answerable: bool
    answer: str
    supporting_chunk_ids: list[str]


__all__ = [
    "GeneratedPolicyAnswer",
    "PolicyAnswerResponse",
    "PolicyAnswerStatus",
    "PolicyCitation",
    "PolicyQuestionRequest",
]
