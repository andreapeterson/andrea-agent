"""Integration helpers for external systems."""

from .embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingRequestError,
    EmbeddingResponseError,
    OpenAIEmbeddingProvider,
)
from .handoff_client import (
    HandoffClient,
    HandoffClientError,
    HandoffNotRequiredError,
    HandoffRequestError,
    HandoffResponseError,
)
from .legacy_crm import LegacyCRMClient, LegacyCRMParseError, LegacyCRMRequestError, parse_customer_xml
from .policy_answer_generator import (
    OpenAIPolicyAnswerGenerator,
    PolicyAnswerGenerator,
    PolicyGenerationError,
    PolicyGenerationRequestError,
    PolicyGenerationResponseError,
    build_policy_answer_prompt,
)
from .scheduler_client import (
    CallerConfirmationRequiredError,
    IdempotencyConflictError,
    SchedulingConflictError,
    SchedulerClient,
    SchedulerError,
    SchedulerRequestError,
    SchedulerResponseError,
    SlotNotFoundError,
    SlotUnavailableError,
)

__all__ = [
    "CallerConfirmationRequiredError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingRequestError",
    "EmbeddingResponseError",
    "HandoffClient",
    "HandoffClientError",
    "HandoffNotRequiredError",
    "HandoffRequestError",
    "HandoffResponseError",
    "IdempotencyConflictError",
    "LegacyCRMClient",
    "LegacyCRMParseError",
    "LegacyCRMRequestError",
    "OpenAIEmbeddingProvider",
    "OpenAIPolicyAnswerGenerator",
    "PolicyAnswerGenerator",
    "PolicyGenerationError",
    "PolicyGenerationRequestError",
    "PolicyGenerationResponseError",
    "SchedulingConflictError",
    "SchedulerClient",
    "SchedulerError",
    "SchedulerRequestError",
    "SchedulerResponseError",
    "SlotNotFoundError",
    "SlotUnavailableError",
    "build_policy_answer_prompt",
    "parse_customer_xml",
]
