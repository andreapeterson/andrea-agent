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
    "SchedulingConflictError",
    "SchedulerClient",
    "SchedulerError",
    "SchedulerRequestError",
    "SchedulerResponseError",
    "SlotNotFoundError",
    "SlotUnavailableError",
    "parse_customer_xml",
]
