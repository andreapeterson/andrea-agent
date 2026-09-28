"""Domain models for PawLine."""

from .appointment import (
    AppointmentSlot,
    AppointmentType,
    BookingConfirmation,
    BookingRequest,
    BookingStatus,
)
from .conversation import ConversationMessage, ConversationPhase, ConversationRole, ConversationState
from .customer import Customer, Pet, PetSpecies
from .handoff import HandoffCreateRequest, HandoffReceipt, HandoffRequest, HandoffStatus, HandoffSummary
from .policy import PolicyChunk, PolicyDocument
from .policy_answer import (
    GeneratedPolicyAnswer,
    PolicyAnswerResponse,
    PolicyAnswerStatus,
    PolicyCitation,
    PolicyQuestionRequest,
)
from .retrieval import EmbeddedPolicyChunk, PolicySearchResult
from .routing import (
    IntakeAnswers,
    RoutingAction,
    RoutingDecision,
    RoutingLevel,
)

__all__ = [
    "AppointmentSlot",
    "AppointmentType",
    "BookingConfirmation",
    "BookingRequest",
    "BookingStatus",
    "ConversationMessage",
    "ConversationPhase",
    "ConversationRole",
    "ConversationState",
    "Customer",
    "EmbeddedPolicyChunk",
    "HandoffCreateRequest",
    "HandoffReceipt",
    "HandoffRequest",
    "HandoffStatus",
    "HandoffSummary",
    "IntakeAnswers",
    "Pet",
    "PolicyAnswerResponse",
    "PolicyAnswerStatus",
    "PolicyChunk",
    "PolicyCitation",
    "PolicyDocument",
    "PolicyQuestionRequest",
    "PolicySearchResult",
    "PetSpecies",
    "GeneratedPolicyAnswer",
    "RoutingAction",
    "RoutingDecision",
    "RoutingLevel",
]
