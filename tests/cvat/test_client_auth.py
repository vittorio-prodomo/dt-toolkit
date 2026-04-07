"""Tests for CvatClient authentication path selection."""

from unittest.mock import MagicMock, patch

import pytest


def test_token_path_uses_access_token_credentials(monkeypatch):
    """When DT_CVAT_TOKEN is set, login() receives AccessTokenCredentials."""
    monkeypatch.setenv("DT_CVAT_URL", "http://localhost:8080")
    monkeypatch.setenv("DT_CVAT_TOKEN", "test-token-abc123")
    monkeypatch.delenv("DT_CVAT_USERNAME", raising=False)
    monkeypatch.delenv("DT_CVAT_PASSWORD", raising=False)
    monkeypatch.delenv("DT_CVAT_ORGANIZATION", raising=False)

    with patch("cvat_sdk.core.client.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from cvat_sdk.core.client import AccessTokenCredentials
        from detection_toolkit.cvat.client import CvatClient

        CvatClient()

        mock_client.login.assert_called_once()
        creds = mock_client.login.call_args.kwargs["credentials"]
        assert isinstance(creds, AccessTokenCredentials)
        assert creds.token == "test-token-abc123"


def test_username_password_path_uses_tuple_credentials(monkeypatch):
    """Without DT_CVAT_TOKEN, login() receives a (username, password) tuple."""
    monkeypatch.setenv("DT_CVAT_URL", "http://localhost:8080")
    monkeypatch.delenv("DT_CVAT_TOKEN", raising=False)
    monkeypatch.setenv("DT_CVAT_USERNAME", "alice")
    monkeypatch.setenv("DT_CVAT_PASSWORD", "test-password")
    monkeypatch.delenv("DT_CVAT_ORGANIZATION", raising=False)

    with patch("cvat_sdk.core.client.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        from detection_toolkit.cvat.client import CvatClient

        CvatClient()

        mock_client.login.assert_called_once()
        creds = mock_client.login.call_args.kwargs["credentials"]
        assert creds == ("alice", "test-password")


def test_no_credentials_raises_value_error(monkeypatch):
    """Without any credentials, CvatClient raises ValueError mentioning both options."""
    monkeypatch.setenv("DT_CVAT_URL", "http://localhost:8080")
    monkeypatch.delenv("DT_CVAT_TOKEN", raising=False)
    monkeypatch.delenv("DT_CVAT_USERNAME", raising=False)
    monkeypatch.delenv("DT_CVAT_PASSWORD", raising=False)
    monkeypatch.delenv("DT_CVAT_ORGANIZATION", raising=False)

    from detection_toolkit.cvat.client import CvatClient

    with pytest.raises(ValueError) as excinfo:
        CvatClient()

    msg = str(excinfo.value)
    assert "DT_CVAT_TOKEN" in msg
    assert "DT_CVAT_USERNAME" in msg or "DT_CVAT_PASSWORD" in msg
