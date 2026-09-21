from pathlib import Path

import pytest

from app.integrations.legacy_crm import LegacyCRMParseError, parse_customer_xml
from app.models.customer import PetSpecies

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def test_parse_customer_valid_with_two_pets() -> None:
    xml_text = _read_fixture("customer_found.xml")

    customer = parse_customer_xml(xml_text)

    assert customer.customer_id == "cust_1001"
    assert customer.first_name == "Andrea"
    assert customer.last_name == "Peterson"
    assert customer.phone_number == "3212222222"
    assert len(customer.pets) == 2
    assert customer.pets[0].pet_id == "pet_2001"
    assert customer.pets[0].name == "Milo"
    assert customer.pets[0].species == PetSpecies.DOG
    assert customer.pets[1].pet_id == "pet_2002"
    assert customer.pets[1].name == "Maize"
    assert customer.pets[1].species == PetSpecies.DOG


def test_parse_customer_valid_with_no_pets() -> None:
    xml_text = _read_fixture("customer_no_pets.xml")

    customer = parse_customer_xml(xml_text)

    assert customer.customer_id == "cust_2001"
    assert customer.pets == []


def test_parse_customer_xml_malformed() -> None:
    xml_text = "<customer><customerId>cust_1001</customerId>"

    with pytest.raises(LegacyCRMParseError):
        parse_customer_xml(xml_text)


def test_parse_customer_xml_missing_required_customer_field() -> None:
    xml_text = """
    <customer>
      <firstName>Andrea</firstName>
      <lastName>Peterson</lastName>
      <phoneNumber>3212222222</phoneNumber>
      <pets />
    </customer>
    """

    with pytest.raises(LegacyCRMParseError):
        parse_customer_xml(xml_text)


def test_parse_customer_xml_unsupported_pet_species() -> None:
    xml_text = """
    <customer>
      <customerId>cust_1001</customerId>
      <firstName>Andrea</firstName>
      <lastName>Peterson</lastName>
      <phoneNumber>3212222222</phoneNumber>
      <pets>
        <pet>
          <petId>pet_2001</petId>
          <name>Luna</name>
          <species>bird</species>
        </pet>
      </pets>
    </customer>
    """

    with pytest.raises(LegacyCRMParseError):
        parse_customer_xml(xml_text)
