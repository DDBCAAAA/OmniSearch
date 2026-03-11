"""Application settings loaded from environment variables.

All credentials and runtime configuration must be provided via environment
variables (or a local `.env` file in development). Do not hardcode secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Holds global application settings.

    Attributes:
        gcp_project_id: Google Cloud project ID.
        gcp_region: Google Cloud region for Vertex AI.
        postgres_host: PostgreSQL host.
        postgres_port: PostgreSQL port.
        postgres_db: PostgreSQL database name.
        postgres_user: PostgreSQL user.
        postgres_password: PostgreSQL password.
    """

    gcp_project_id: str
    gcp_region: str
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str


def load_settings() -> Settings:
    """Load settings from environment variables.

    Returns:
        A populated Settings instance.

    Raises:
        KeyError: If a required environment variable is missing.
        ValueError: If a value cannot be parsed (e.g., non-integer port).
    """
    return Settings(
        gcp_project_id=os.environ["GCP_PROJECT_ID"],
        gcp_region=os.environ.get("GCP_REGION", "us-central1"),
        postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
        postgres_db=os.environ.get("POSTGRES_DB", "omnisearch"),
        postgres_user=os.environ.get("POSTGRES_USER", "postgres"),
        postgres_password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )


__all__ = ["Settings", "load_settings"]

