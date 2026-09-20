import os

from fastapi.testclient import TestClient

os.environ["MOCK_CRM_API_KEY"] = "dev-crm-key"

from mock_services.legacy_crm_api import app


client = TestClient(app)


def test_customer_lookup_known_phone_returns_200() -> None:
    response = client.get(
        "/customers/by-phone",
        params={"phone": "3212222222"},
        headers={"X-API-Key": "dev-crm-key"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "cust_1001" in response.text
    assert "Andrea" in response.text
    assert "Maize" in response.text
    assert "pet_2001" in response.text
    assert "Milo" in response.text


def test_customer_lookup_missing_api_key_returns_401() -> None:
    response = client.get(
        "/customers/by-phone",
        params={"phone": "3212222222"},
    )

    assert response.status_code == 401

def test_customer_lookup_incorrect_api_key_returns_401() -> None:
    response = client.get(
        "/customers/by-phone",
        params={"phone": "3212222222"},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401

def test_missing_phone_returns_422() -> None:
    response = client.get(
        "/customers/by-phone",
        headers={"X-API-Key": "dev-crm-key"},
    )

    assert response.status_code == 422

def test_customer_lookup_unknown_phone_returns_404() -> None:
    response = client.get(
        "/customers/by-phone",
        params={"phone": "9995550000"},
        headers={"X-API-Key": "dev-crm-key"},
    )

    assert response.status_code == 404
