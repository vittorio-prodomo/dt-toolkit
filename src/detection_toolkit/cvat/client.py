"""Thin connection wrapper around cvat_sdk with env-var defaults."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class CvatClient:
    """Connect to a CVAT server using explicit args or ``DT_CVAT_*`` env vars.

    The env vars (typically in the project ``.env``) are:

    * ``DT_CVAT_URL`` — server URL (default ``http://localhost:8080``)
    * ``DT_CVAT_TOKEN`` — Personal Access Token (preferred when set)
    * ``DT_CVAT_USERNAME`` — used with password as fallback if no token
    * ``DT_CVAT_PASSWORD`` — used with username as fallback if no token
    * ``DT_CVAT_ORGANIZATION`` — optional workspace slug

    Authentication preference: if a token is provided (explicit ``token``
    arg or ``DT_CVAT_TOKEN`` env var), the client logs in via cvat-sdk's
    ``AccessTokenCredentials``. Otherwise it falls back to username/password.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        organization: Optional[str] = None,
        status_check_period: int = 5,
    ) -> None:
        from cvat_sdk.core.client import AccessTokenCredentials, Client, Config

        self._url = url or os.environ.get("DT_CVAT_URL", "http://localhost:8080")
        self._token = token or os.environ.get("DT_CVAT_TOKEN", "")
        self._username = username or os.environ.get("DT_CVAT_USERNAME", "")
        self._password = password or os.environ.get("DT_CVAT_PASSWORD", "")
        self._organization = organization or os.environ.get("DT_CVAT_ORGANIZATION")

        if not self._token and not (self._username and self._password):
            raise ValueError(
                "CVAT credentials required. Set DT_CVAT_TOKEN (preferred) "
                "or DT_CVAT_USERNAME/DT_CVAT_PASSWORD in .env, or pass "
                "--token / --username and --password."
            )

        if self._token:
            logger.info("Connecting to CVAT at %s via PAT", self._url)
            credentials = AccessTokenCredentials(self._token)
        else:
            logger.info("Connecting to CVAT at %s as %s", self._url, self._username)
            credentials = (self._username, self._password)

        self._client = Client(
            url=self._url,
            config=Config(status_check_period=status_check_period),
        )
        self._client.login(credentials=credentials)
        if self._organization:
            self._client.organization_slug = self._organization
        logger.info("Connected")

    @property
    def sdk(self):
        """The underlying ``cvat_sdk.core.client.Client``."""
        return self._client
