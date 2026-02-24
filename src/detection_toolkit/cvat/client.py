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
    * ``DT_CVAT_USERNAME``
    * ``DT_CVAT_PASSWORD``
    * ``DT_CVAT_ORGANIZATION`` — optional workspace slug
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        organization: Optional[str] = None,
        status_check_period: int = 5,
    ) -> None:
        from cvat_sdk.core.client import Client, Config

        self._url = url or os.environ.get("DT_CVAT_URL", "http://localhost:8080")
        self._username = username or os.environ.get("DT_CVAT_USERNAME", "")
        self._password = password or os.environ.get("DT_CVAT_PASSWORD", "")
        self._organization = organization or os.environ.get("DT_CVAT_ORGANIZATION")

        if not self._username or not self._password:
            raise ValueError(
                "CVAT credentials required. Set DT_CVAT_USERNAME / DT_CVAT_PASSWORD "
                "in .env or pass --username / --password."
            )

        logger.info("Connecting to CVAT at %s as %s", self._url, self._username)
        self._client = Client(
            url=self._url,
            config=Config(status_check_period=status_check_period),
        )
        self._client.login(credentials=(self._username, self._password))
        if self._organization:
            self._client.organization_slug = self._organization
        logger.info("Connected")

    @property
    def sdk(self):
        """The underlying ``cvat_sdk.core.client.Client``."""
        return self._client
