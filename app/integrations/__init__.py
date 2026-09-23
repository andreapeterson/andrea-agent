"""Integration helpers for external systems."""

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
