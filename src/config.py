"""Application configuration for the text-ml-platform project.

This module centralises runtime configuration so that scripts can be
configured in one place (or via environment variables) instead of
hard-coding values. All modules should import from here rather than
calling os.getenv directly.
"""

from __future__ import annotations

import os
from pathlib import Path


def _get_env_list(name: str, default: str) -> list[str]:
    """Parse a comma-separated env var into a list."""
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Kafka configuration
# ---------------------------------------------------------------------------

# Bootstrap servers for connecting to Kafka.
# Default matches the docker-compose single-broker setup.
KAFKA_BOOTSTRAP_SERVERS: list[str] = _get_env_list(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092",
)

# Topic used for streaming IMDb reviews into the platform.
KAFKA_IMDB_TOPIC: str = os.getenv("KAFKA_IMDB_TOPIC", "imdb-reviews")

# Topic for async UI → inference_worker only. Separate from KAFKA_IMDB_TOPIC so bulk
# prepopulate train/test traffic does not sit ahead of inference in the same partition log.
KAFKA_INFERENCE_TOPIC: str = os.getenv("KAFKA_INFERENCE_TOPIC", "imdb-inference")

# Dead-letter topic for failed messages (e.g., async inference failures).
KAFKA_IMDB_DLQ_TOPIC: str = os.getenv("KAFKA_IMDB_DLQ_TOPIC", "imdb-reviews-dlq")


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
# Iceberg catalog
# ---------------------------------------------------------------------------


def _iceberg_catalog_db_default() -> str:
    """Default path for SQLite catalog DB (project root / iceberg_catalog / catalog.db)."""
    project_root = Path(__file__).resolve().parents[1]  # src/ -> project root
    default_dir = project_root / "iceberg_catalog"
    default_dir.mkdir(parents=True, exist_ok=True)
    return str(default_dir / "catalog.db")


ICEBERG_CATALOG_DB: str = os.getenv("ICEBERG_CATALOG_DB", _iceberg_catalog_db_default())


# ---------------------------------------------------------------------------
# IMDb dataset settings
# ---------------------------------------------------------------------------

# Default split to use when streaming IMDb reviews.
IMDB_DEFAULT_SPLIT: str = os.getenv("IMDB_DEFAULT_SPLIT", "test")

# Optional hard cap to avoid loading the full 50k record dataset in small demos.
IMDB_MAX_RECORDS: int | None = int(os.getenv("IMDB_MAX_RECORDS", "")) if os.getenv("IMDB_MAX_RECORDS") else None


# ---------------------------------------------------------------------------
# Demo / UI configuration
# ---------------------------------------------------------------------------

PREDICT_API_URL: str = os.getenv("PREDICT_API_URL", "http://localhost:8000/predict")
GOLD_ICEBERG_IDENTIFIER: str = os.getenv("GOLD_ICEBERG_IDENTIFIER", "imdb.gold_train")
PCA_MAX_POINTS: int = int(os.getenv("PCA_MAX_POINTS", "1500"))
OLLAMA_API_BASE: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")

# Default classifier model path (relative to project root).
DEFAULT_MODEL_PATH: str = os.getenv("DEFAULT_MODEL_PATH", "models/sentiment_logreg.joblib")

# Async prediction: how long the UI waits for the worker to write to imdb.predictions (seconds).
ASYNC_PREDICTION_TIMEOUT_S: int = int(os.getenv("ASYNC_PREDICTION_TIMEOUT_S", "180"))


__all__ = [
    "ASYNC_PREDICTION_TIMEOUT_S",
    "BRONZE_PREFIX",
    "DEFAULT_MODEL_PATH",
    "FEATURES_PREFIX",
    "GOLD_ICEBERG_IDENTIFIER",
    "GOLD_PREFIX",
    "ICEBERG_CATALOG_DB",
    "IMDB_DEFAULT_SPLIT",
    "IMDB_MAX_RECORDS",
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_IMDB_DLQ_TOPIC",
    "KAFKA_IMDB_TOPIC",
    "KAFKA_INFERENCE_TOPIC",
    "OLLAMA_API_BASE",
    "PCA_MAX_POINTS",
    "PREDICT_API_URL",
    "S3_ACCESS_KEY",
    "S3_DATA_BUCKET",
    "S3_ENDPOINT_URL",
    "S3_REGION_NAME",
    "S3_SECRET_KEY",
    "SILVER_PREFIX",
]
