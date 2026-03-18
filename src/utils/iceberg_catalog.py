"""Iceberg catalog utilities for gold layer.

Uses PyIceberg SqlCatalog with SQLite metadata and S3/MinIO warehouse.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from src import config

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog


def _warehouse_uri() -> str:
    """S3 warehouse URI for MinIO."""
    endpoint = (config.S3_ENDPOINT_URL or "http://localhost:9000").replace("http://", "").replace("https://", "")
    bucket = config.S3_DATA_BUCKET
    return f"s3://{bucket}/iceberg"


def _catalog_db_path() -> str:
    """Path to SQLite catalog DB (local). Uses project root so CLI and notebooks share the same catalog."""
    project_root = Path(__file__).resolve().parents[2]
    default_dir = project_root / "iceberg_catalog"
    default_dir.mkdir(parents=True, exist_ok=True)
    return os.getenv("ICEBERG_CATALOG_DB", str(default_dir / "catalog.db"))


def get_iceberg_catalog() -> Catalog:
    """Return a configured Iceberg catalog for MinIO.

    Uses SqlCatalog with SQLite for metadata and S3 warehouse.
    """
    from pyiceberg.catalog.sql import SqlCatalog

    return SqlCatalog(
        "default",
        **{
            "uri": f"sqlite:///{_catalog_db_path()}",
            "warehouse": _warehouse_uri(),
            "py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO",
            "s3.endpoint": config.S3_ENDPOINT_URL or "http://localhost:9000",
            "s3.access-key-id": config.S3_ACCESS_KEY or "admin",
            "s3.secret-access-key": config.S3_SECRET_KEY or "password123",
            "s3.region": config.S3_REGION_NAME or "us-east-1",
        },
    )


__all__ = ["get_iceberg_catalog"]
