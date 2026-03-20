"""Iceberg catalog utilities for gold layer.

Uses PyIceberg SqlCatalog with SQLite metadata and S3/MinIO warehouse.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from src import config

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog


def _warehouse_uri() -> str:
    """S3 warehouse URI for MinIO."""
    return f"s3://{config.S3_DATA_BUCKET}/iceberg"


@functools.lru_cache(maxsize=1)
def get_iceberg_catalog() -> Catalog:
    """Return a configured Iceberg catalog for MinIO.

    Uses SqlCatalog with SQLite for metadata and S3 warehouse.
    The catalog is cached so all callers share a single instance.
    """
    from pyiceberg.catalog.sql import SqlCatalog

    return SqlCatalog(
        "default",
        **{
            "uri": f"sqlite:///{config.ICEBERG_CATALOG_DB}",
            "warehouse": _warehouse_uri(),
            # Prefer PyArrowFileIO for MinIO/S3 compatibility in containers.
            # This avoids s3fs credential kwarg collisions with mixed environments.
            "py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO",
            "s3.endpoint": config.S3_ENDPOINT_URL or "http://localhost:9000",
            "s3.region": config.S3_REGION_NAME or "us-east-1",
            # Pass credentials directly to PyIceberg/S3FileSystem to avoid
            # ambiguous merging of AWS_* and S3_* environment variables.
            "s3.access-key-id": config.S3_ACCESS_KEY or "",
            "s3.secret-access-key": config.S3_SECRET_KEY or "",
        },
    )


__all__ = ["get_iceberg_catalog"]
