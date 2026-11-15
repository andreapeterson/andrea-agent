"""Keep application credential retrieval offline unless a test supplies its fake."""

from unittest.mock import Mock

import pytest


@pytest.fixture(autouse=True)
def mock_runtime_secret_lookup(monkeypatch):
    monkeypatch.setattr("app.dependencies.get_openai_api_key", Mock(return_value=None))
