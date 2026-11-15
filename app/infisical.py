"""Load the OpenAI credential from Infisical without exposing secret data."""

import os
import sys
from contextlib import suppress
from functools import lru_cache

from infisical_sdk import InfisicalError, InfisicalSDKClient


@lru_cache(maxsize=1)
def get_openai_api_key() -> str:
    """Fetch once per process; restart the app after rotating the OpenAI key."""
    names = ("INFISICAL_CLIENT_ID", "INFISICAL_CLIENT_SECRET", "INFISICAL_PROJECT_ID")
    config = {name: os.getenv(name, "").strip() for name in names}
    missing = [name for name in names if not config[name]]
    if missing:
        raise RuntimeError("Missing Infisical configuration: " + ", ".join(missing))

    stage = "client initialization"
    client = None
    failure_hint = ""
    try:
        client = InfisicalSDKClient(host="https://app.infisical.com")
        stage = "authentication"
        client.auth.universal_auth.login(
            client_id=config["INFISICAL_CLIENT_ID"],
            client_secret=config["INFISICAL_CLIENT_SECRET"],
        )
        stage = "secret lookup"
        secret = client.secrets.get_secret_by_name(
            secret_name="OPENAI_API_KEY",
            project_id=config["INFISICAL_PROJECT_ID"],
            environment_slug="dev",
            secret_path="/",
        )
        stage = "secret validation"
        value = secret.secretValue
        if (
            isinstance(value, str) and value.strip()
            and value.strip() != "<hidden-by-infisical>"
        ):
            return value
    except (InfisicalError, OSError, RuntimeError, TypeError, ValueError, AttributeError) as error:
        # Copy only class/status metadata; never stringify or retain the SDK error.
        status = getattr(error, "status_code", None)
        status_hint = f", HTTP {status}" if type(status) is int and 100 <= status <= 599 else ""
        failure_hint = f" ({type(error).__name__}{status_hint})"
    finally:
        if client is not None:
            with suppress(Exception):
                client.api.session.close()
                client.close()
    raise RuntimeError(f"Infisical retrieval failed during {stage}{failure_hint}.")


def main() -> int:
    """Verify retrieval only; do not construct or call an OpenAI client."""
    try:
        get_openai_api_key()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    print("OK: retrieved the configured OpenAI credential from Infisical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
