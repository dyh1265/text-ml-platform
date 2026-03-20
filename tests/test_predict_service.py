"""Tests for the prediction API (FastAPI TestClient)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def _mock_deps():
    """Patch heavy dependencies so tests run without GPU / model files."""
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    fake_bert = MagicMock()
    fake_tokenizer = MagicMock()

    import torch

    fake_enc = [torch.zeros(1, 768)]

    def fake_load_classifier(_model_path):
        return fake_model, {}

    with (
        patch("src.inference.predict_service._load_classifier", side_effect=fake_load_classifier),
        patch("src.inference.predict_service.AutoModel") as mock_auto,
        patch("src.inference.predict_service.AutoTokenizer") as mock_tok,
        patch("src.inference.predict_service.encode_batch", return_value=fake_enc),
    ):
        mock_auto.from_pretrained.return_value = fake_bert
        mock_tok.from_pretrained.return_value = fake_tokenizer
        yield


@pytest.fixture()
def client(_mock_deps):
    from pathlib import Path

    from fastapi.testclient import TestClient

    from src.inference.predict_service import create_app

    app = create_app(
        classifier_model_path=Path("models/sentiment_logreg.joblib"),
        bert_model_name="bert-base-uncased",
        finetuned_bert_path=None,
        device="cpu",
        iceberg_write=False,
    )
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json={"text": "This movie was fantastic!"})
    assert resp.status_code == 200
    body = resp.json()
    assert "pred_label" in body
    assert "pred_score" in body
    assert "request_id" in body
    assert body["pred_label"] in (0, 1)
    assert 0.0 <= body["pred_score"] <= 1.0


def test_predict_empty_text(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pred_label"] == 0
    assert body["pred_score"] == 0.0


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
