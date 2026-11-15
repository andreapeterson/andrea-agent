"""Verify the existing helper with a fake SDK and dummy credentials."""

import traceback
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from app import infisical


@pytest.fixture
def sdk(monkeypatch):
    infisical.get_openai_api_key.cache_clear()
    for name, value in {
        "INFISICAL_CLIENT_ID": "dummy-client",
        "INFISICAL_CLIENT_SECRET": "dummy-bootstrap-secret",
        "INFISICAL_PROJECT_ID": "demo-project",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENAI_API_KEY", "unused-environment-key")
    client = MagicMock()
    client.secrets.get_secret_by_name.return_value = SimpleNamespace(secretValue="dummy-openai-key")
    factory = Mock(return_value=client)
    monkeypatch.setattr(infisical, "InfisicalSDKClient", factory)
    yield SimpleNamespace(client=client, factory=factory)
    infisical.get_openai_api_key.cache_clear()


def test_existing_scope_authentication_and_process_cache(sdk, caplog, capsys):
    assert infisical.get_openai_api_key() == "dummy-openai-key"
    assert infisical.get_openai_api_key() == "dummy-openai-key"
    sdk.factory.assert_called_once_with(host="https://app.infisical.com")
    sdk.client.auth.universal_auth.login.assert_called_once_with(
        client_id="dummy-client", client_secret="dummy-bootstrap-secret",
    )
    sdk.client.secrets.get_secret_by_name.assert_called_once_with(
        secret_name="OPENAI_API_KEY", project_id="demo-project",
        environment_slug="dev", secret_path="/",
    )
    sdk.client.api.session.close.assert_called_once()
    sdk.client.close.assert_called_once()
    assert caplog.text == ""
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("name", ["INFISICAL_CLIENT_ID", "INFISICAL_CLIENT_SECRET", "INFISICAL_PROJECT_ID"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_only_three_bootstrap_variables_are_required(sdk, monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=f"Missing Infisical configuration: {name}"):
        infisical.get_openai_api_key()
    sdk.factory.assert_not_called()


@pytest.mark.parametrize("value", [None, "", "  ", "<hidden-by-infisical>", 123])
def test_blank_masked_or_malformed_values_do_not_fall_back_to_environment(sdk, value):
    sdk.client.secrets.get_secret_by_name.return_value.secretValue = value
    with pytest.raises(RuntimeError, match="secret validation"):
        infisical.get_openai_api_key()
    sdk.client.close.assert_called_once()


@pytest.mark.parametrize("stage", ["client initialization", "authentication", "secret lookup"])
def test_sdk_errors_are_sanitized_and_can_be_retried(sdk, stage, caplog, capsys):
    private = "dummy-bootstrap-secret dummy-openai-key raw-response demo-project"
    call = {
        "client initialization": sdk.factory,
        "authentication": sdk.client.auth.universal_auth.login,
        "secret lookup": sdk.client.secrets.get_secret_by_name,
    }[stage]
    call.side_effect = RuntimeError(private)
    with pytest.raises(RuntimeError) as error:
        infisical.get_openai_api_key()
    assert str(error.value) == f"Infisical retrieval failed during {stage} (RuntimeError)."
    assert error.value.__context__ is None
    assert private not in "".join(traceback.format_exception(error.value)) + caplog.text
    assert capsys.readouterr() == ("", "")
    call.side_effect = None
    assert infisical.get_openai_api_key() == "dummy-openai-key"


def test_standalone_smoke_check_prints_only_safe_confirmation_or_error(sdk, capsys):
    assert infisical.main() == 0
    assert capsys.readouterr() == (
        "OK: retrieved the configured OpenAI credential from Infisical.\n", "",
    )
    infisical.get_openai_api_key.cache_clear()
    sdk.client.secrets.get_secret_by_name.side_effect = RuntimeError("PRIVATE_RESPONSE")
    assert infisical.main() == 1
    assert capsys.readouterr() == ("", "Infisical retrieval failed during secret lookup (RuntimeError).\n")


@pytest.mark.parametrize("stage", ["authentication", "secret lookup"])
@pytest.mark.parametrize("status", [None, 403, 404, True, "PRIVATE_RESPONSE"])
def test_diagnostics_expose_only_stage_class_and_valid_http_status(sdk, stage, status, capsys):
    class UpstreamError(RuntimeError):
        def __str__(self):
            raise AssertionError("Never stringify an upstream exception.")

        def __repr__(self):
            raise AssertionError("Never represent an upstream exception.")

    error = UpstreamError("PRIVATE_BOOTSTRAP PRIVATE_PROJECT PRIVATE_SECRET PRIVATE_BODY")
    error.status_code = status
    error.response = {"body": "PRIVATE_RESPONSE"}
    call = (
        sdk.client.auth.universal_auth.login if stage == "authentication"
        else sdk.client.secrets.get_secret_by_name
    )
    call.side_effect = error
    assert infisical.main() == 1
    status_hint = f", HTTP {status}" if type(status) is int else ""
    assert capsys.readouterr() == (
        "", f"Infisical retrieval failed during {stage} (UpstreamError{status_hint}).\n",
    )
