import os

from app.integrations import LegacyCRMClient


def get_legacy_crm_client() -> LegacyCRMClient:
    base_url = os.getenv("LEGACY_CRM_BASE_URL", "http://127.0.0.1:8001")
    api_key = os.getenv("LEGACY_CRM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LEGACY_CRM_API_KEY is not configured. Set it in the environment before starting the app."
        )
    return LegacyCRMClient(base_url=base_url, api_key=api_key)
