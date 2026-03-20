"""Prediction API for the demo.

Exposes a small FastAPI service:
- accepts an input review text
- cleans it
- embeds it with BERT
- runs the trained classifier
- (optionally) writes the prediction to an Iceberg table for lineage.
"""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Any

import joblib
import pyarrow as pa
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification

from src.features.embedding_job import encode_batch
from src.transformation.cleaning import clean_text
from src.utils.iceberg_catalog import get_iceberg_catalog
from src.utils.metrics import (
    PREDICT_LABEL_TOTAL,
    PREDICT_LATENCY_SECONDS,
    PREDICT_REQUESTS_TOTAL,
    get_metrics_body,
)
from src.utils.structured_logging import log_event


class PredictRequest(BaseModel):
    text: str
    request_id: str | None = None
    split: str = "inference"


class PredictResponse(BaseModel):
    request_id: str
    pred_label: int
    pred_score: float
    cleaned_text: str


def _resolve_model_path(model_path: Path) -> Path:
    """Resolve model path; if relative, interpret relative to project root."""
    if model_path.is_absolute():
        return model_path
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / model_path).resolve()


def _load_classifier(model_path: Path) -> tuple[Any, dict[str, Any]]:
    resolved = _resolve_model_path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Model file not found: {resolved}\n"
            "Train the classifier first, e.g.:\n"
            "  python -m src.training.train_classifier "
            "--iceberg-identifier imdb.gold_train --model-out models/sentiment_logreg.joblib"
        )
    obj = joblib.load(resolved)
    if "model" not in obj:
        raise ValueError(f"Unexpected model artifact format: {model_path}")
    return obj["model"], obj.get("metrics", {})


def write_prediction_to_iceberg(
    *,
    request_id: str,
    text: str,
    cleaned_text: str,
    pred_label: int,
    pred_score: float,
    namespace: str = "imdb",
    table_name: str = "predictions",
    model_version: str = "logreg",
) -> None:
    """Append a single prediction row to an Iceberg table."""
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    catalog = get_iceberg_catalog()
    catalog.create_namespace_if_not_exists(namespace)

    schema = Schema(
        NestedField(1, "request_id", StringType(), required=True),
        NestedField(2, "text", StringType(), required=True),
        NestedField(3, "cleaned_text", StringType(), required=True),
        NestedField(4, "pred_label", LongType(), required=True),
        NestedField(5, "pred_score", DoubleType(), required=True),
        NestedField(6, "model_version", StringType(), required=True),
        NestedField(7, "created_at_ms", LongType(), required=True),
        NestedField(8, "split", StringType(), required=True),
    )

    identifier = f"{namespace}.{table_name}"
    from pyiceberg.exceptions import NoSuchTableError

    try:
        tbl = catalog.load_table(identifier)
    except NoSuchTableError:
        tbl = catalog.create_table(identifier, schema=schema)

    now_ms = int(time.time() * 1000)
    arrow_schema = pa.schema(
        [
            pa.field("request_id", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("cleaned_text", pa.string(), nullable=False),
            pa.field("pred_label", pa.int64(), nullable=False),
            pa.field("pred_score", pa.float64(), nullable=False),
            pa.field("model_version", pa.string(), nullable=False),
            pa.field("created_at_ms", pa.int64(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
        ]
    )
    arrow_table = pa.table(
        {
            "request_id": [request_id],
            "text": [text],
            "cleaned_text": [cleaned_text],
            "pred_label": [int(pred_label)],
            "pred_score": [float(pred_score)],
            "model_version": [model_version],
            "created_at_ms": [now_ms],
            "split": ["inference"],
        },
        schema=arrow_schema,
    )
    tbl.append(arrow_table)


def _predict_with_finetuned(model, tokenizer, cleaned: str, device_obj) -> tuple[int, float]:
    """Run prediction with fine-tuned BertForSequenceClassification."""
    enc = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(device_obj) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits, dim=1)[0]
    pred_score = float(probs[1])
    pred_label = int(pred_score >= 0.5)
    return pred_label, pred_score


def create_app(
    *,
    classifier_model_path: Path,
    bert_model_name: str,
    finetuned_bert_path: Path | None,
    device: str,
    iceberg_write: bool,
) -> FastAPI:
    app = FastAPI(title="text-ml-platform prediction service (demo)")

    device_obj = torch.device(device)
    use_finetuned = (
        finetuned_bert_path is not None
        and finetuned_bert_path.is_dir()
        and (finetuned_bert_path / "config.json").exists()
    )
    if finetuned_bert_path and not use_finetuned:
        log_event("finetuned_bert_not_found", path=str(finetuned_bert_path), level="warning")

    if use_finetuned:
        tokenizer = AutoTokenizer.from_pretrained(str(finetuned_bert_path))
        model = BertForSequenceClassification.from_pretrained(str(finetuned_bert_path)).to(device_obj)
        model.eval()
        app.state.clf = None
        app.state.metrics = {}
        app.state.model = model
        app.state.use_finetuned = True
    else:
        clf, metrics = _load_classifier(classifier_model_path)
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert_model = AutoModel.from_pretrained(bert_model_name).to(device_obj)
        bert_model.eval()
        app.state.clf = clf
        app.state.metrics = metrics
        app.state.bert_model = bert_model
        app.state.model = None
        app.state.use_finetuned = False

    app.state.tokenizer = tokenizer
    app.state.device = device_obj
    app.state.iceberg_write = iceberg_write

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        PREDICT_REQUESTS_TOTAL.labels(split=req.split).inc()
        start = time.perf_counter()
        request_id = req.request_id or str(uuid.uuid4())

        raw_text = req.text or ""
        cleaned = clean_text(raw_text)
        log_event("predict_received", request_id=request_id, split=req.split, cleaned_nonempty=bool(cleaned))
        if not cleaned:
            PREDICT_LABEL_TOTAL.labels(label="0").inc()
            PREDICT_LATENCY_SECONDS.observe(time.perf_counter() - start)
            return PredictResponse(request_id=request_id, pred_label=0, pred_score=0.0, cleaned_text="")

        if app.state.use_finetuned:
            pred_label, pred_score = _predict_with_finetuned(
                app.state.model, app.state.tokenizer, cleaned, app.state.device
            )
        else:
            with torch.no_grad():
                enc = encode_batch(app.state.bert_model, app.state.tokenizer, [cleaned], app.state.device)
            X = enc[0].cpu().numpy().astype("float32").reshape(1, -1)

            if hasattr(app.state.clf, "predict_proba"):
                proba = app.state.clf.predict_proba(X)[0]
                pred_score = float(proba[1])
                pred_label = int(pred_score >= 0.5)
            else:
                pred_label = int(app.state.clf.predict(X)[0])
                pred_score = float(pred_label)

        PREDICT_LABEL_TOTAL.labels(label=str(pred_label)).inc()

        if app.state.iceberg_write:
            try:
                write_prediction_to_iceberg(
                    request_id=request_id,
                    text=raw_text,
                    cleaned_text=cleaned,
                    pred_label=pred_label,
                    pred_score=pred_score,
                )
            except Exception as exc:
                # Best-effort only: log and continue serving the prediction.
                log_event("predict_iceberg_write_failed", request_id=request_id, error=str(exc))

        PREDICT_LATENCY_SECONDS.observe(time.perf_counter() - start)
        log_event(
            "predict_completed",
            request_id=request_id,
            pred_label=pred_label,
            pred_score=pred_score,
        )
        return PredictResponse(
            request_id=request_id,
            pred_label=pred_label,
            pred_score=pred_score,
            cleaned_text=cleaned,
        )

    @app.get("/metrics")
    def metrics_endpoint():
        """Prometheus metrics endpoint for scraping."""
        return Response(content=get_metrics_body(), media_type="text/plain; charset=utf-8")

    @app.get("/health")
    def health():
        return {"ok": True, "metrics": app.state.metrics}

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediction API service (demo).")
    parser.add_argument("--model-path", type=str, default="models/sentiment_logreg.joblib")
    parser.add_argument("--bert-model", type=str, default="textattack/bert-base-uncased-SST-2")
    parser.add_argument(
        "--finetuned-bert-path",
        type=str,
        default=None,
        help="Path to fine-tuned BERT dir; when set, used for inference (no LogReg).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iceberg-write", action="store_true", help="Write predictions to Iceberg (imdb.predictions).")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        p = project_root / path_str
    return p.resolve()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    finetuned = _resolve_path(args.finetuned_bert_path) if args.finetuned_bert_path else None
    app = create_app(
        classifier_model_path=Path(args.model_path),
        bert_model_name=args.bert_model,
        finetuned_bert_path=finetuned,
        device=args.device,
        iceberg_write=args.iceberg_write,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
