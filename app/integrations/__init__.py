"""Integration helpers for external systems."""

from .legacy_crm import LegacyCRMClient, LegacyCRMParseError, LegacyCRMRequestError, parse_customer_xml

__all__ = [
    "LegacyCRMClient",
    "LegacyCRMParseError",
    "LegacyCRMRequestError",
    "parse_customer_xml",
]
