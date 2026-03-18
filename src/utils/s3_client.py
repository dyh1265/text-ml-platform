"""S3 client utilities for talking to MinIO/S3."""

from __future__ import annotations

import time
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src import config

_UPLOAD_RETRIES = 3
_UPLOAD_RETRY_DELAY = 2.0


def get_s3_client():
    """Return a configured boto3 S3 client for MinIO or AWS S3."""
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=config.S3_ENDPOINT_URL,
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY,
        region_name=config.S3_REGION_NAME,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket_exists(bucket: str | None = None) -> None:
    """Create the data bucket if it does not already exist."""
    bucket = bucket or config.S3_DATA_BUCKET
    client = get_s3_client()

    existing = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if bucket in existing:
        return

    client.create_bucket(Bucket=bucket)


def upload_bytes(key: str, data: bytes, bucket: str | None = None) -> None:
    """Upload raw bytes to the configured bucket under the given key.
    Retries on AccessDenied and other transient errors.
    """
    bucket = bucket or config.S3_DATA_BUCKET
    client = get_s3_client()
    last_error = None
    for attempt in range(_UPLOAD_RETRIES):
        try:
            client.put_object(Bucket=bucket, Key=key, Body=data)
            return
        except ClientError as e:
            last_error = e
            if attempt < _UPLOAD_RETRIES - 1:
                time.sleep(_UPLOAD_RETRY_DELAY)
    raise last_error


def list_objects(prefix: str, bucket: str | None = None) -> list[dict]:
    """List objects under prefix. Returns list of {'Key': str, 'Size': int, ...}."""
    bucket = bucket or config.S3_DATA_BUCKET
    client = get_s3_client()
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return resp.get("Contents", [])


def get_object_body(key: str, bucket: str | None = None) -> bytes:
    """Download object and return raw bytes."""
    bucket = bucket or config.S3_DATA_BUCKET
    client = get_s3_client()
    resp = client.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


__all__ = [
    "get_s3_client",
    "ensure_bucket_exists",
    "upload_bytes",
    "list_objects",
    "get_object_body",
]
