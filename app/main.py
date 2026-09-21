from fastapi import Depends, FastAPI, HTTPException, Query

from app.dependencies import get_legacy_crm_client
from app.integrations import LegacyCRMParseError, LegacyCRMRequestError
from app.models.customer import Customer

app = FastAPI(title="PawLine")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/customers/lookup", response_model=Customer)
async def lookup_customer(
    phone: str = Query(..., description="Customer phone number."),
    crm_client=Depends(get_legacy_crm_client), #constructionc of client separate. route only for customer lookup. also lets test replace client with fake client.
) -> Customer:
    try:
        customer = await crm_client.find_customer_by_phone(phone)
    except LegacyCRMRequestError as error:
        raise HTTPException(status_code=503, detail="customer-service-unavailable") from error
    except LegacyCRMParseError as error:
        raise HTTPException(status_code=502, detail="invalid-upstream-data") from error

    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    return customer

