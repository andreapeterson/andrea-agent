"""Load the OpenAI credential from Infisical without exposing secret data."""

import os
import sys
from functools import lru_cache

from infisical_sdk import InfisicalSDKClient


@lru_cache(maxsize=1)
def get_openai_api_key() -> str:
    """Fetch once per process; restart the app after rotating the OpenAI key."""
    names = ("INFISICAL_CLIENT_ID", "INFISICAL_CLIENT_SECRET", "INFISICAL_PROJECT_ID")
    config = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name in names if not config[name]]
    if missing:
        raise RuntimeError("Missing Infisical configuration: " + ", ".join(missing))

    stage = "client initialization"
    try:
        # Initialize the client
        client = InfisicalSDKClient(host="https://app.infisical.com")
        # Authenticate (example using Universal Auth)
        stage = "authentication"
        client.auth.universal_auth.login(
            client_id=config["INFISICAL_CLIENT_ID"],
            client_secret=config["INFISICAL_CLIENT_SECRET"],
        )
        # Use the SDK to interact with Infisical
        stage = "secret lookup"
        secret = client.secrets.get_secret_by_name(
            secret_name="AGENT_API_KEY",
            project_id=config["INFISICAL_PROJECT_ID"],
            environment_slug="dev",
            secret_path="/",
        )
        stage = "secret validation"
        api_key = secret.secretValue
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or api_key.strip() == "<hidden-by-infisical>"
        ):
            raise ValueError("Empty or masked secret")
    except Exception as error:
        # SDK errors may contain response bodies. Never expose the original error.
        status = getattr(error, "status_code", None)
        status_hint = f" (HTTP {status})" if isinstance(status, int) and 100 <= status <= 599 else ""
        raise RuntimeError(
            f"Infisical retrieval failed during {stage}{status_hint}. "
            "Check local credentials, project access, "
            "and AGENT_API_KEY in dev at /."
        ) from None

    return api_key


def main() -> int:
    """Verify retrieval only; do not construct or call an OpenAI client."""
    try:
        get_openai_api_key()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    print("OK: retrieved a non-empty AGENT_API_KEY from Infisical (dev, /).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
