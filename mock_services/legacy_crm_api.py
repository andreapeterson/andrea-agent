import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import Response

app = FastAPI(title="Mock Legacy Veterinary CRM")

DATA_FILE = Path(__file__).resolve().parent / "data" / "customer_3215550100.xml"


def _expected_api_key() -> str:
    return os.getenv("MOCK_CRM_API_KEY", "dev-crm-key")


@app.get("/customers/by-phone")
def get_customer_by_phone(
    phone: str = Query(..., description="Customer phone number."),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Response:
    if x_api_key is None or x_api_key != _expected_api_key():
        raise HTTPException(status_code=401, detail="Unauthorized")

    if phone != "3212222222":
        raise HTTPException(status_code=404, detail="Customer not found")

    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Customer not found")

    xml_content = DATA_FILE.read_text(encoding="utf-8")
    return Response(content=xml_content, media_type="application/xml")
