import os

from app.integrations import HandoffClient, LegacyCRMClient, SchedulerClient


def get_legacy_crm_client() -> LegacyCRMClient:
    base_url = os.getenv("LEGACY_CRM_BASE_URL", "http://127.0.0.1:8001")
    api_key = os.getenv("LEGACY_CRM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "LEGACY_CRM_API_KEY is not configured. Set it in the environment before starting the app."
        )
    return LegacyCRMClient(base_url=base_url, api_key=api_key)


def get_scheduler_client() -> SchedulerClient:
    base_url = os.getenv("SCHEDULER_BASE_URL", "http://127.0.0.1:8002")
    jwt_secret = os.getenv("SCHEDULER_JWT_SECRET")
    if not jwt_secret:
        raise RuntimeError(
            "SCHEDULER_JWT_SECRET is not configured. Set it in the environment before starting the app."
        )
    return SchedulerClient(base_url=base_url, jwt_secret=jwt_secret)


def get_handoff_client() -> HandoffClient:
    base_url = os.getenv("HANDOFF_BASE_URL", "http://127.0.0.1:8003")
    webhook_secret = os.getenv("HANDOFF_WEBHOOK_SECRET")
    if not webhook_secret:
        raise RuntimeError(
            "HANDOFF_WEBHOOK_SECRET is not configured. Set it in the environment before starting the app."
        )
    return HandoffClient(base_url=base_url, webhook_secret=webhook_secret)
