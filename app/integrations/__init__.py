"""Integration helpers for external systems."""

from .legacy_crm import LegacyCRMParseError, parse_customer_xml

__all__ = ["LegacyCRMParseError", "parse_customer_xml"]
