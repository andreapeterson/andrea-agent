import xml.etree.ElementTree as ET

import httpx

from app.models.customer import Customer, Pet, PetSpecies


class LegacyCRMParseError(ValueError):
    """Raised when the legacy CRM XML payload is invalid."""


class LegacyCRMRequestError(RuntimeError):
    """Raised when the legacy CRM HTTP request fails."""


def _require_text(element: ET.Element, tag: str, label: str) -> str:
    child = element.find(tag)
    if child is None or child.text is None or not child.text.strip():
        raise LegacyCRMParseError(f"Missing required {label} field: {tag}")
    return child.text.strip()


def parse_customer_xml(xml_text: str) -> Customer:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise LegacyCRMParseError(f"Malformed XML: {exc}") from exc

    customer_id = _require_text(root, "customerId", "customer")
    first_name = _require_text(root, "firstName", "customer")
    last_name = _require_text(root, "lastName", "customer")
    phone_number = _require_text(root, "phoneNumber", "customer")

    pets: list[Pet] = []
    pets_root = root.find("pets")
    if pets_root is not None:
        for pet_element in pets_root.findall("pet"):
            pet_id = _require_text(pet_element, "petId", "pet")
            name = _require_text(pet_element, "name", "pet")
            species_text = _require_text(pet_element, "species", "pet")

            try:
                species = PetSpecies(species_text)
            except ValueError as exc:
                raise LegacyCRMParseError(f"Unsupported pet species: {species_text}") from exc

            pets.append(Pet(pet_id=pet_id, name=name, species=species))

    return Customer(
        customer_id=customer_id,
        first_name=first_name,
        last_name=last_name,
        phone_number=phone_number,
        pets=pets,
    )


class LegacyCRMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 5.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def find_customer_by_phone(self, phone: str) -> Customer | None:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                response = await client.get(
                    "/customers/by-phone",
                    params={"phone": phone},
                    headers={"X-API-Key": self._api_key},
                )
            except httpx.RequestError as exc:
                raise LegacyCRMRequestError("Legacy CRM request failed.") from exc

        if response.status_code == 200:
            return parse_customer_xml(response.text)
        if response.status_code == 404:
            return None

        raise LegacyCRMRequestError(f"Legacy CRM request failed with status {response.status_code}.")
