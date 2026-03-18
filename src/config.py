"""Application configuration for the text-ml-platform project.

This module centralises runtime configuration so that scripts can be
configured in one place (or via environment variables) instead of
hard-coding values.
"""

from __future__ import annotations

import os
from typing import List


def _get_env_list(name: str, default: str) -> List[str]:
    """Parse a comma-separated env var into a list."""
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Kafka configuration
# ---------------------------------------------------------------------------

# Bootstrap servers for connecting to Kafka.
# Default matches the docker-compose single-broker setup.
KAFKA_BOOTSTRAP_SERVERS: List[str] = _get_env_list(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

# Topic used for streaming IMDb reviews into the platform.
KAFKA_IMDB_TOPIC: str = os.getenv("KAFKA_IMDB_TOPIC", "imdb-reviews")


# ---------------------------------------------------------------------------
# Object storage (MinIO/S3-compatible) configuration
# ---------------------------------------------------------------------------

S3_ENDPOINT_URL: str | None = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY: str | None = os.getenv("S3_ACCESS_KEY", "admin")
S3_SECRET_KEY: str | None = os.getenv("S3_SECRET_KEY", "password123")
S3_REGION_NAME: str | None = os.getenv("S3_REGION_NAME", "us-east-1")

# Bucket where bronze/silver/feature data is stored.
S3_DATA_BUCKET: str = os.getenv("S3_DATA_BUCKET", "text-ml-data")

# Paths inside the bucket for different layers.
BRONZE_PREFIX: str = os.getenv("BRONZE_PREFIX", "bronze/")
SILVER_PREFIX: str = os.getenv("SILVER_PREFIX", "silver/")
GOLD_PREFIX: str = os.getenv("GOLD_PREFIX", "gold/")
FEATURES_PREFIX: str = os.getenv("FEATURES_PREFIX", "features/")


# ---------------------------------------------------------------------------
# IMDb dataset settings
# ---------------------------------------------------------------------------

# Default split to use when streaming IMDb reviews.
IMDB_DEFAULT_SPLIT: str = os.getenv("IMDB_DEFAULT_SPLIT", "test")

# Optional hard cap to avoid loading the full 50k record dataset in small demos.
IMDB_MAX_RECORDS: int | None = (
    int(os.getenv("IMDB_MAX_RECORDS", "")) if os.getenv("IMDB_MAX_RECORDS") else None
)


__all__ = [
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_IMDB_TOPIC",
    "S3_ENDPOINT_URL",
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    "S3_REGION_NAME",
    "S3_DATA_BUCKET",
    "BRONZE_PREFIX",
    "SILVER_PREFIX",
    "GOLD_PREFIX",
    "FEATURES_PREFIX",
    "IMDB_DEFAULT_SPLIT",
    "IMDB_MAX_RECORDS",
]
