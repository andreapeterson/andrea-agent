from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_legacy_crm_client
from app.integrations import LegacyCRMParseError, LegacyCRMRequestError
from app.main import app
from app.models.customer import Customer, Pet, PetSpecies


class FakeCRMClient:
    def __init__(self) -> None:
        self.phone_numbers: list[str] = []
        self.customer_to_return: Customer | None = None
        self.should_raise_request_error: bool = False
        self.should_raise_parse_error: bool = False

    async def find_customer_by_phone(self, phone: str) -> Customer | None:
        self.phone_numbers.append(phone)

        if self.should_raise_request_error:
            raise LegacyCRMRequestError("fake request failure")
        if self.should_raise_parse_error:
            raise LegacyCRMParseError("fake parse failure")
        return self.customer_to_return


@pytest.fixture
def fake_crm_client() -> FakeCRMClient:
    return FakeCRMClient()


@pytest.fixture
def client(fake_crm_client: FakeCRMClient) -> TestClient:
    app.dependency_overrides[get_legacy_crm_client] = lambda: fake_crm_client
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_customer_lookup_valid_customer(client: TestClient, fake_crm_client: FakeCRMClient) -> None:
    fake_crm_client.customer_to_return = Customer(
        customer_id="cust_1001",
        first_name="Andrea",
        last_name="Peterson",
        phone_number="3212222222",
        pets=[
            Pet(pet_id="pet_2001", name="Milo", species=PetSpecies.DOG),
            Pet(pet_id="pet_2002", name="Maize", species=PetSpecies.DOG),
        ],
    )

    response = client.get("/customers/lookup?phone=3215550100")

    assert response.status_code == 200
    payload = response.json()
    assert payload["customer_id"] == "cust_1001"
    assert payload["first_name"] == "Andrea"
    assert payload["last_name"] == "Peterson"
    assert payload["phone_number"] == "3212222222"
    assert payload["pets"][0]["pet_id"] == "pet_2001"
    assert payload["pets"][0]["name"] == "Milo"
    assert payload["pets"][0]["species"] == "dog"
    assert payload["pets"][1]["pet_id"] == "pet_2002"
    assert payload["pets"][1]["name"] == "Maize"
    assert payload["pets"][1]["species"] == "dog"
    assert fake_crm_client.phone_numbers == ["3215550100"]


def test_customer_lookup_customer_not_found(client: TestClient, fake_crm_client: FakeCRMClient) -> None:
    fake_crm_client.customer_to_return = None

    response = client.get("/customers/lookup?phone=9999999999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Customer not found"


def test_customer_lookup_crm_request_failure(client: TestClient, fake_crm_client: FakeCRMClient) -> None:
    fake_crm_client.should_raise_request_error = True

    response = client.get("/customers/lookup?phone=3215550100")

    assert response.status_code == 503
    assert response.json()["detail"] == "customer-service-unavailable"
    assert "fake request failure" not in response.text


def test_customer_lookup_invalid_upstream_data(client: TestClient, fake_crm_client: FakeCRMClient) -> None:
    fake_crm_client.should_raise_parse_error = True

    response = client.get("/customers/lookup?phone=3215550100")

    assert response.status_code == 502
    assert response.json()["detail"] == "invalid-upstream-data"
    assert "fake parse failure" not in response.text


def test_customer_lookup_missing_phone(client: TestClient) -> None:
    response = client.get("/customers/lookup")

    assert response.status_code == 422
