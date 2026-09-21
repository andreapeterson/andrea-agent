from pathlib import Path

import httpx
import pytest

from app.integrations import LegacyCRMClient, LegacyCRMParseError, LegacyCRMRequestError

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def read_valid_customer_xml() -> str:
    return (FIXTURE_DIR / "customer_found.xml").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_find_customer_by_phone_success() -> None:
    xml_text = read_valid_customer_xml()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/customers/by-phone"
        assert request.url.params["phone"] == "3212222222"
        assert request.headers["X-API-Key"] == "dev-crm-key"
        return httpx.Response(200, text=xml_text, request=request)

    client = LegacyCRMClient(
        base_url="https://legacy-crm.example.com/",
        api_key="dev-crm-key",
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    customer = await client.find_customer_by_phone("3212222222")

    assert customer is not None
    assert customer.customer_id == "cust_1001"
    assert customer.first_name == "Andrea"
    assert customer.last_name == "Peterson"
    assert customer.phone_number == "3212222222"
    assert customer.pets[0].name == "Milo"


@pytest.mark.asyncio
async def test_find_customer_by_phone_not_found_returns_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/customers/by-phone"
        assert request.url.params["phone"] == "9999999999"
        assert request.headers["X-API-Key"] == "dev-crm-key"
        return httpx.Response(404, text="Not Found", request=request)

    client = LegacyCRMClient(
        base_url="https://legacy-crm.example.com",
        api_key="dev-crm-key",
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    customer = await client.find_customer_by_phone("9999999999")

    assert customer is None


@pytest.mark.asyncio
async def test_find_customer_by_phone_unauthorized_raises_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized", request=request)

    client = LegacyCRMClient(
        base_url="https://legacy-crm.example.com",
        api_key="dev-crm-key",
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LegacyCRMRequestError, match="Legacy CRM request failed with status 401"):
        await client.find_customer_by_phone("3212222222")


@pytest.mark.asyncio
async def test_find_customer_by_phone_server_error_raises_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Server Error", request=request)

    client = LegacyCRMClient(
        base_url="https://legacy-crm.example.com",
        api_key="dev-crm-key",
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LegacyCRMRequestError, match="Legacy CRM request failed with status 500"):
        await client.find_customer_by_phone("3212222222")


@pytest.mark.asyncio
async def test_find_customer_by_phone_connection_error_raises_request_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated connection failure", request=request)

    client = LegacyCRMClient(
        base_url="https://legacy-crm.example.com",
        api_key="dev-crm-key",
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LegacyCRMRequestError, match="Legacy CRM request failed"):
        await client.find_customer_by_phone("3212222222")


@pytest.mark.asyncio
async def test_find_customer_by_phone_malformed_xml_raises_parse_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<customer><customerId>broken</customerId>", request=request)

    client = LegacyCRMClient(
        base_url="https://legacy-crm.example.com",
        api_key="dev-crm-key",
        timeout_seconds=2.0,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LegacyCRMParseError):
        await client.find_customer_by_phone("3212222222")
