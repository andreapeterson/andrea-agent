"""Integration helpers for external systems."""

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
    "HandoffClient",
    "HandoffClientError",
    "HandoffNotRequiredError",
    "HandoffRequestError",
    "HandoffResponseError",
    "IdempotencyConflictError",
    "LegacyCRMClient",
    "LegacyCRMParseError",
    "LegacyCRMRequestError",
    "SchedulingConflictError",
    "SchedulerClient",
    "SchedulerError",
    "SchedulerRequestError",
    "SchedulerResponseError",
    "SlotNotFoundError",
    "SlotUnavailableError",
    "parse_customer_xml",
]
