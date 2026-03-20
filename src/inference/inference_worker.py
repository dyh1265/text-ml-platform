"""Kafka consumer worker for async inference (demo).

Consumes inference requests from Kafka, and for each request:
- writes bronze + silver records (so the UI can visualize lineage)
- embeds the text and writes to an Iceberg gold table
- runs the trained classifier and appends a row to an Iceberg predictions table

This worker is designed for the demo / production-feel; it is not optimized
for throughput.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import uuid
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import torch
from kafka import KafkaConsumer, KafkaProducer
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification

from src import config
from src.features.embedding_job import encode_batch
from src.inference.predict_service import write_prediction_to_iceberg
from src.transformation.cleaning import clean_text
from src.utils.iceberg_catalog import get_iceberg_catalog
from src.utils.metrics import (
    INFERENCE_CONSUMED_TOTAL,
    INFERENCE_FAILURE_TOTAL,
    INFERENCE_PROCESSING_SECONDS,
    INFERENCE_SUCCESS_TOTAL,
)
from src.utils.s3_client import ensure_bucket_exists, upload_bytes
from src.utils.schema import ImdbBronzeReview, ImdbSilverReview
from src.utils.structured_logging import log_event


def _write_bronze_and_silver(
    *,
    request_id: str,
    record_id: str,
    raw_text: str,
    cleaned_text: str,
    label: int,
    split: str,
    bronze_out_prefix: str,
    silver_out_prefix: str,
) -> None:
    timestamp = int(time.time())

    bronze_key = f"{bronze_out_prefix}imdb_bronze_{timestamp}_{uuid.uuid4().hex}.jsonl"
    bronze_review = ImdbBronzeReview(
        id=record_id,
        text=raw_text,
        label=label,
        split=split,  # "inference"
        request_id=request_id,
    )
    bronze_line = json.dumps(bronze_review.to_message(), ensure_ascii=False)
    upload_bytes(bronze_key, (bronze_line + "\n").encode("utf-8"))

    silver_key = f"{silver_out_prefix}imdb_silver_{timestamp}_{uuid.uuid4().hex}.jsonl"
    silver_review = ImdbSilverReview(
        id=record_id,
        text=cleaned_text,
        label=label,
        split=split,
        request_id=request_id,
    )
    silver_line = json.dumps(silver_review.to_dict(), ensure_ascii=False)
    upload_bytes(silver_key, (silver_line + "\n").encode("utf-8"))


def _write_gold_iceberg(
    *,
    request_id: str,
    record_id: str,
    text: str,
    label: int,
    split: str,
    embedding_vec: list[float],
    iceberg_namespace: str,
    iceberg_table: str,
) -> None:
    """Append one gold embedding row to an Iceberg table."""
    from pyiceberg.exceptions import NoSuchTableError

    from src.utils.schema import gold_embedding_arrow_schema, gold_embedding_iceberg_schema

    catalog = get_iceberg_catalog()
    catalog.create_namespace_if_not_exists(iceberg_namespace)

    schema = gold_embedding_iceberg_schema()
    identifier = f"{iceberg_namespace}.{iceberg_table}"
    try:
        tbl = catalog.load_table(identifier)
    except NoSuchTableError:
        tbl = catalog.create_table(identifier, schema=schema)

    embedding_f32 = np.array(embedding_vec, dtype=np.float32).tolist()
    arrow_table = pa.table(
        {
            "id": [record_id],
            "text": [text],
            "label": [int(label)],
            "embedding": [embedding_f32],
            "split": [split],
            "request_id": [request_id],
        },
        schema=gold_embedding_arrow_schema(),
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


def run_worker(
    *,
    iceberg_namespace: str,
    iceberg_table: str,
    classifier_model_path: Path,
    bert_model_name: str,
    finetuned_bert_path: Path | None,
    device: str,
    iceberg_write: bool,
    from_kafka_split: str = "inference",
    bronze_prefix: str | None = None,
    silver_prefix: str | None = None,
) -> None:
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
        full_model = BertForSequenceClassification.from_pretrained(str(finetuned_bert_path)).to(device_obj)
        full_model.eval()
        bert_model = full_model.bert  # for embeddings
        clf = None
    else:
        clf_obj = joblib.load(classifier_model_path)
        clf = clf_obj["model"]
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert_model = AutoModel.from_pretrained(bert_model_name).to(device_obj)
        bert_model.eval()

    bronze_prefix = bronze_prefix or f"{config.BRONZE_PREFIX}imdb/"
    silver_prefix = silver_prefix or f"{config.SILVER_PREFIX}imdb/"

    # We still write under split-aware paths.
    bronze_out_prefix = bronze_prefix + "inference/"
    silver_out_prefix = silver_prefix + "inference/"

    ensure_bucket_exists()

    consumer = KafkaConsumer(
        config.KAFKA_INFERENCE_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        group_id="imdb-inference-worker",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    dlq_producer: KafkaProducer | None = None
    try:
        dlq_producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    except Exception:
        log_event("dlq_producer_init_failed", level="warning")

    log_event("inference_worker_started", topic=config.KAFKA_INFERENCE_TOPIC)
    for msg in consumer:
        proc_start = time.perf_counter()
        try:
            payload = msg.value
            msg_split = payload.get("split", from_kafka_split)
            if msg_split != from_kafka_split:
                continue

            INFERENCE_CONSUMED_TOTAL.inc()
            request_id = payload.get("request_id") or payload.get("id") or str(uuid.uuid4())
            record_id = payload.get("id") or request_id
            raw_text = str(payload.get("text", ""))
            label = int(payload.get("label") or 0)

            cleaned = clean_text(raw_text)
            if not cleaned:
                pred_label, pred_score = 0, 0.0
                X = None
            else:
                if use_finetuned:
                    pred_label, pred_score = _predict_with_finetuned(full_model, tokenizer, cleaned, device_obj)
                    with torch.no_grad():
                        enc = encode_batch(bert_model, tokenizer, [cleaned], device_obj)
                    X = enc[0].cpu().numpy().astype("float32").reshape(1, -1)
                else:
                    with torch.no_grad():
                        enc = encode_batch(bert_model, tokenizer, [cleaned], device_obj)
                    X = enc[0].cpu().numpy().astype("float32").reshape(1, -1)
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(X)[0]
                        pred_score = float(proba[1])
                        pred_label = int(pred_score >= 0.5)
                    else:
                        pred_label = int(clf.predict(X)[0])
                        pred_score = float(pred_label)

            # Write lineage stages.
            _write_bronze_and_silver(
                request_id=request_id,
                record_id=str(record_id),
                raw_text=raw_text,
                cleaned_text=cleaned,
                label=label,
                split=from_kafka_split,
                bronze_out_prefix=bronze_out_prefix,
                silver_out_prefix=silver_out_prefix,
            )

            # Write prediction first (critical for UI). Gold is best-effort and may fail if MinIO is full.
            for attempt in range(3):
                try:
                    write_prediction_to_iceberg(
                        request_id=str(request_id),
                        text=raw_text,
                        cleaned_text=cleaned,
                        pred_label=int(pred_label),
                        pred_score=float(pred_score),
                        namespace=iceberg_namespace,
                        table_name="predictions",
                    )
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)

            if iceberg_write:
                try:
                    emb_vec = X[0].tolist() if cleaned and X is not None else [0.0] * 768
                    _write_gold_iceberg(
                        request_id=request_id,
                        record_id=str(record_id),
                        text=cleaned,
                        label=label,
                        split=from_kafka_split,
                        embedding_vec=emb_vec,
                        iceberg_namespace=iceberg_namespace,
                        iceberg_table=iceberg_table,
                    )
                except Exception as gold_exc:
                    log_event(
                        "gold_inference_write_failed", request_id=str(request_id), error=str(gold_exc), level="warning"
                    )
            INFERENCE_SUCCESS_TOTAL.inc()
            INFERENCE_PROCESSING_SECONDS.observe(time.perf_counter() - proc_start)
            log_event(
                "inference_completed",
                request_id=str(request_id),
                pred_label=int(pred_label),
                pred_score=float(pred_score),
            )
        except Exception as exc:
            INFERENCE_FAILURE_TOTAL.inc()
            INFERENCE_PROCESSING_SECONDS.observe(time.perf_counter() - proc_start)
            log_event(
                "inference_failed",
                request_id=request_id,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            log_event("inference_failed_detail", level="error", request_id=request_id, traceback=traceback.format_exc())
            if dlq_producer is not None:
                try:
                    dlq_producer.send(
                        config.KAFKA_IMDB_DLQ_TOPIC,
                        {
                            "error": str(exc),
                            "ts_ms": int(time.time() * 1000),
                            "payload": msg.value,
                        },
                    )
                    dlq_producer.flush()
                except Exception:
                    pass

            log_event("inference_failed", error=str(exc))
            continue


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async inference Kafka worker (demo).")
    parser.add_argument("--classifier-model", type=str, default="models/sentiment_logreg.joblib")
    parser.add_argument("--bert-model", type=str, default="textattack/bert-base-uncased-SST-2")
    parser.add_argument(
        "--finetuned-bert-path",
        type=str,
        default=None,
        help="Path to fine-tuned BERT dir; when set, used for inference (no LogReg).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--iceberg", action="store_true", help="Also write embeddings to Iceberg.")
    parser.add_argument("--iceberg-namespace", type=str, default="imdb")
    parser.add_argument("--iceberg-table", type=str, default="gold_inference")
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
    run_worker(
        iceberg_namespace=args.iceberg_namespace,
        iceberg_table=args.iceberg_table,
        classifier_model_path=Path(args.classifier_model),
        bert_model_name=args.bert_model,
        finetuned_bert_path=finetuned,
        device=args.device,
        iceberg_write=args.iceberg,
    )


if __name__ == "__main__":
    main()
