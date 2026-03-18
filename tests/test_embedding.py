"""Tests for embedding job."""

from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.features.embedding_job import load_silver_records


def test_load_silver_records_empty():
    """load_silver_records returns empty list when no objects."""
    with patch("src.features.embedding_job.list_objects", return_value=[]):
        result = load_silver_records("silver/imdb/")
    assert result == []


def test_load_silver_records_parses_jsonl():
    """load_silver_records parses JSONL and returns list of dicts."""
    content = b'{"id":"1","text":"great","label":1}\n{"id":"2","text":"bad","label":0}\n'
    with patch("src.features.embedding_job.list_objects") as mock_list:
        with patch("src.features.embedding_job.get_object_body", return_value=content):
            mock_list.return_value = [{"Key": "silver/imdb/file.jsonl"}]
            result = load_silver_records("silver/imdb/")
    assert len(result) == 2
    assert result[0]["id"] == "1" and result[0]["label"] == 1
    assert result[1]["id"] == "2" and result[1]["label"] == 0
