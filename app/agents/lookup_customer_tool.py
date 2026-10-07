"""Define the front-desk customer lookup tool and its CRM-backed Python handler."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from app.integrations.legacy_crm import LegacyCRMClient


class LookupCustomerArguments(BaseModel):
    """Validate the phone number the LLM extracted for a customer lookup.

    The OpenAI response provides tool arguments as JSON. PawLine converts that
    JSON into this model before sending the phone number to LegacyCRMClient.
    This model does not call the CRM itself.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    phone: str = Field(
        ...,
        min_length=1,
        description="Phone number supplied by the caller.",
    )

#tool definition describes the capability to the LLM
LOOKUP_CUSTOMER_TOOL: dict[str, object] = {
    "type": "function",
    "name": "lookup_customer",
    "description": (
        "Call this tool when the caller needs account-specific help and has supplied a phone number. "
        "Use the caller-provided phone number exactly as supplied; do not invent or modify a missing phone number. "
        "Do not use this tool for general clinic-policy questions. "
        "A not-found result is not a system failure. "
        "Do not claim the customer was found until the tool returns found=true."
    ),
    "strict": True,
    "parameters": LookupCustomerArguments.model_json_schema(),
}


def create_lookup_customer_handler(
    crm_client: LegacyCRMClient,
) -> Callable[[dict[str, object]], Awaitable[object]]:
    """Create the real Python function that handles lookup_customer requests.

    The returned handler validates the LLM's arguments and asks the existing
    LegacyCRMClient to find the customer. It returns only the customer details
    the front-desk agent needs for its next response.
    """

    async def handler(arguments: dict[str, object]) -> object:
        validated = LookupCustomerArguments.model_validate(arguments)
        customer = await crm_client.find_customer_by_phone(validated.phone)
        if customer is None:
            return {"found": False}

        return {
            "found": True,
            "first_name": customer.first_name,
            "pets": [
                {
                    "pet_id": pet.pet_id,
                    "name": pet.name,
                }
                for pet in customer.pets
            ],
        }

    return handler


__all__ = ["LOOKUP_CUSTOMER_TOOL", "LookupCustomerArguments", "create_lookup_customer_handler"]