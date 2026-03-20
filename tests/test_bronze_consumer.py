"""Tests for bronze_consumer."""

from unittest.mock import patch

from src.ingestion.bronze_consumer import _flush_batch


def test_flush_batch():
    """_flush_batch uploads JSONL to S3."""
    lines = ['{"id":"1","text":"hi","label":1,"split":"train"}']
    with (
        patch("src.ingestion.bronze_consumer.ensure_bucket_exists"),
        patch("src.ingestion.bronze_consumer.upload_bytes") as mock_upload,
    ):
        _flush_batch(lines, "train")
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args[0]
        assert "imdb_bronze_" in call_args[0]
        assert "train" in call_args[0]
        assert call_args[1].decode().strip().endswith('"split":"train"}')
