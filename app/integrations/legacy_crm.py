import xml.etree.ElementTree as ET

from app.models.customer import Customer, Pet, PetSpecies


class LegacyCRMParseError(ValueError):
    """Raised when the legacy CRM XML payload is invalid."""


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
