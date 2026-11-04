"""Verify the credential boundary with dummy data and no network requests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import dependencies, infisical


@pytest.fixture
def sdk(monkeypatch):
    infisical.get_openai_api_key.cache_clear()
    monkeypatch.setenv("INFISICAL_CLIENT_ID", "dummy-client-id")
    monkeypatch.setenv("INFISICAL_CLIENT_SECRET", "dummy-client-secret")
    monkeypatch.setenv("INFISICAL_PROJECT_ID", "dummy-project-id")
    # An old local key must never override or bypass Infisical.
    monkeypatch.setenv("OPENAI_API_KEY", "unused-local-key")
    client = Mock()
    client.auth.universal_auth.login.return_value = SimpleNamespace(accessToken="dummy-token")
    client.secrets.get_secret_by_name.return_value = SimpleNamespace(secretValue="dummy-openai-key")
    factory = Mock(return_value=client)
    monkeypatch.setattr(infisical, "InfisicalSDKClient", factory)
    yield SimpleNamespace(client=client, factory=factory)
    infisical.get_openai_api_key.cache_clear()


def test_authentication_retrieval_and_process_cache(sdk, capsys):
    assert infisical.get_openai_api_key() == "dummy-openai-key"
    assert infisical.get_openai_api_key() == "dummy-openai-key"
    sdk.factory.assert_called_once_with(host="https://app.infisical.com")
    sdk.client.auth.universal_auth.login.assert_called_once_with(
        client_id="dummy-client-id", client_secret="dummy-client-secret",
    )
    sdk.client.secrets.get_secret_by_name.assert_called_once_with(
        secret_name="AGENT_API_KEY", project_id="dummy-project-id",
        environment_slug="dev", secret_path="/",
    )
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("name", ["INFISICAL_CLIENT_ID", "INFISICAL_CLIENT_SECRET", "INFISICAL_PROJECT_ID"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_configuration_does_not_contact_infisical(sdk, monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=name):
        infisical.get_openai_api_key()
    sdk.factory.assert_not_called()


@pytest.mark.parametrize("stage", ["login", "retrieval"])
def test_failures_are_redacted_and_can_be_retried(sdk, capsys, stage):
    call = (
        sdk.client.auth.universal_auth.login if stage == "login"
        else sdk.client.secrets.get_secret_by_name
    )
    call.side_effect = RuntimeError("dummy-client-secret dummy-token dummy-openai-key")
    assert infisical.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    failure_stage = "authentication" if stage == "login" else "secret lookup"
    assert output.err == (
        f"Infisical retrieval failed during {failure_stage}. "
        "Check local credentials, project access, "
        "and AGENT_API_KEY in dev at /.\n"
    )
    if stage == "login":
        sdk.client.secrets.get_secret_by_name.assert_not_called()
    call.side_effect = None
    assert infisical.get_openai_api_key() == "dummy-openai-key"


@pytest.mark.parametrize("value", [None, "", "  ", "<hidden-by-infisical>", "  <hidden-by-infisical>  "])
def test_empty_or_masked_secrets_fail_verification(sdk, value, capsys):
    sdk.client.secrets.get_secret_by_name.return_value.secretValue = value
    assert infisical.main() == 1
    assert "Infisical retrieval failed" in capsys.readouterr().err


def test_verification_prints_only_confirmation(sdk, capsys):
    assert infisical.main() == 0
    assert capsys.readouterr() == (
        "OK: retrieved a non-empty AGENT_API_KEY from Infisical (dev, /).\n", "",
    )


def test_http_status_is_reported_without_response_or_credentials(sdk, capsys):
    error = RuntimeError("dummy-client-secret dummy-token dummy-openai-key")
    error.status_code = 403
    error.response = {"secretValue": "dummy-openai-key"}
    sdk.client.secrets.get_secret_by_name.side_effect = error
    assert infisical.main() == 1
    assert capsys.readouterr() == (
        "",
        "Infisical retrieval failed during secret lookup (HTTP 403). "
        "Check local credentials, project access, and AGENT_API_KEY in dev at /.\n",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency", [
    "get_prompt_agent_runtime", "get_policy_agent", "get_policy_retriever", "get_policy_answer_service",
])
async def test_app_fails_closed_when_infisical_fails(monkeypatch, dependency):
    loader = Mock(side_effect=RuntimeError("Infisical retrieval failed"))
    openai = Mock()
    monkeypatch.setattr(dependencies, "get_openai_api_key", loader)
    monkeypatch.setattr(dependencies, "AsyncOpenAI", openai)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert await getattr(dependencies, dependency)(request) is None
    loader.assert_called_once_with()
    openai.assert_not_called()
