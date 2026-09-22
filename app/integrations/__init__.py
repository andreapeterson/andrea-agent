"""Integration helpers for external systems."""

from .legacy_crm import LegacyCRMClient, LegacyCRMParseError, LegacyCRMRequestError, parse_customer_xml
from .scheduler_client import (
    CallerConfirmationRequiredError,
    SchedulerClient,
    SchedulerError,
    SchedulerRequestError,
    SchedulerResponseError,
    SlotNotFoundError,
)

__all__ = [
    "CallerConfirmationRequiredError",
    "LegacyCRMClient",
    "LegacyCRMParseError",
    "LegacyCRMRequestError",
    "SchedulerClient",
    "SchedulerError",
    "SchedulerRequestError",
    "SchedulerResponseError",
    "SlotNotFoundError",
    "parse_customer_xml",
]
