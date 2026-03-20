"""Tests for embedding job."""

from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.features.embedding_job import encode_batch, load_silver_records, run_embedding_job


def test_encode_batch():
    """encode_batch returns pooler_output from mock model."""
    mock_model = MagicMock()
    mock_out = MagicMock()
    mock_out.pooler_output = torch.zeros(2, 768)
    mock_model.return_value = mock_out

    mock_tok = MagicMock()
    mock_tok.return_value = {
        "input_ids": torch.zeros(2, 10, dtype=torch.long),
        "attention_mask": torch.ones(2, 10),
        "token_type_ids": torch.zeros(2, 10, dtype=torch.long),
    }

    device = torch.device("cpu")
    result = encode_batch(mock_model, mock_tok, ["hello", "world"], device)
    assert result.shape == (2, 768)
    mock_model.assert_called_once()


def test_load_silver_records_empty():
    """load_silver_records returns empty list when no objects."""
    with patch("src.features.embedding_job.list_objects", return_value=[]):
        result = load_silver_records("silver/imdb/")
    assert result == []


def test_load_silver_records_parses_jsonl():
    """load_silver_records parses JSONL and returns list of dicts."""
    content = b'{"id":"1","text":"great","label":1}\n{"id":"2","text":"bad","label":0}\n'
    with (
        patch("src.features.embedding_job.list_objects") as mock_list,
        patch("src.features.embedding_job.get_object_body", return_value=content),
    ):
        mock_list.return_value = [{"Key": "silver/imdb/file.jsonl"}]
        result = load_silver_records("silver/imdb/")
    assert len(result) == 2
    assert result[0]["id"] == "1" and result[0]["label"] == 1
    assert result[1]["id"] == "2" and result[1]["label"] == 0


def test_run_embedding_job_parquet_no_silver():
    """run_embedding_job returns 0 when no silver records."""
    with (
        patch("src.features.embedding_job.ensure_bucket_exists"),
        patch("src.features.embedding_job.load_silver_records", return_value=[]),
    ):
        result = run_embedding_job(
            silver_prefix="silver/imdb/",
            gold_prefix="gold/",
            iceberg=False,
        )
    assert result == 0


def test_run_embedding_job_parquet_writes():
    """run_embedding_job encodes records and writes Parquet."""
    records = [
        {"id": "1", "text": "great film", "label": 1, "split": "train", "request_id": None},
        {"id": "2", "text": "terrible", "label": 0, "split": "train", "request_id": None},
    ]
    fake_enc = torch.zeros(2, 768)

    with (
        patch("src.features.embedding_job.ensure_bucket_exists"),
        patch("src.features.embedding_job.load_silver_records", return_value=records),
        patch("src.features.embedding_job._load_bert_for_embedding") as mock_load,
        patch("src.features.embedding_job.upload_bytes"),
    ):
        mock_model = MagicMock()
        mock_tok = MagicMock()
        mock_load.return_value = (mock_model, mock_tok)
        with patch("src.features.embedding_job.encode_batch", return_value=fake_enc):
            result = run_embedding_job(
                silver_prefix="silver/imdb/",
                gold_prefix="gold/",
                iceberg=False,
            )
    assert result == 2
