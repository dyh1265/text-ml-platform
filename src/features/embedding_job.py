"""Embedding generation job.

Reads silver JSONL from MinIO, encodes text with BERT, and writes gold
as Parquet or Iceberg table.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import argparse

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification

from src import config
from src.utils.s3_client import ensure_bucket_exists, get_object_body, list_objects, upload_bytes
from src.utils.structured_logging import log_event

DEFAULT_MODEL = "textattack/bert-base-uncased-SST-2"
DEFAULT_BATCH_SIZE = 32


def load_silver_records(silver_prefix: str) -> list[dict]:
    """Load all silver JSONL records from MinIO under the given prefix."""
    objects = list_objects(silver_prefix)
    records = []
    for obj in objects:
        key = obj["Key"]
        if not key.endswith(".jsonl"):
            continue
        body = get_object_body(key)
        for line in body.decode("utf-8").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def encode_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Encode a batch of texts; returns [batch_size, hidden_size] tensor."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    return out.pooler_output


def _write_gold_parquet(table: pa.Table, gold_prefix: str) -> None:
    """Write gold table as Parquet to MinIO."""
    timestamp = int(time.time())
    out_key = f"{gold_prefix}imdb_gold_{timestamp}.parquet"
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    buf.seek(0)
    upload_bytes(out_key, buf.read())
    log_event("gold_parquet_written", key=out_key, records=table.num_rows)


def _write_gold_iceberg(table: pa.Table, namespace: str = "imdb", table_name: str = "gold") -> None:
    """Write gold table to Iceberg in MinIO."""
    import time

    from pyiceberg.exceptions import NoSuchTableError

    from src.utils.iceberg_catalog import get_iceberg_catalog
    from src.utils.schema import gold_embedding_iceberg_schema

    catalog = get_iceberg_catalog()
    catalog.create_namespace_if_not_exists(namespace)

    schema = gold_embedding_iceberg_schema()
    identifier = f"{namespace}.{table_name}"
    try:
        tbl = catalog.load_table(identifier)
    except NoSuchTableError:
        for create_attempt in range(5):
            try:
                tbl = catalog.create_table(identifier, schema=schema)
                break
            except OSError as exc:
                if ("ACCESS_DENIED" in str(exc) or "CompleteMultipartUpload" in str(exc)) and create_attempt < 4:
                    delay = (create_attempt + 1) * 8
                    log_event(
                        "gold_iceberg_create_retry",
                        attempt=create_attempt + 1,
                        delay_s=delay,
                        error=str(exc)[:200],
                        level="warning",
                    )
                    time.sleep(delay)
                    continue
                raise

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            tbl.append(table)
            break
        except OSError as exc:
            last_err = exc
            if ("ACCESS_DENIED" in str(exc) or "CompleteMultipartUpload" in str(exc)) and attempt < 4:
                delay = (attempt + 1) * 8
                log_event(
                    "gold_iceberg_retry", attempt=attempt + 1, delay_s=delay, error=str(exc)[:200], level="warning"
                )
                time.sleep(delay)
                continue
            raise
        except Exception as exc:
            msg = str(exc)
            is_schema_mismatch = ("Mismatch in fields" in msg) or (
                "schema" in msg.lower() and "mismatch" in msg.lower()
            )
            if not is_schema_mismatch:
                raise
            iceberg_cols = {f.name for f in tbl.schema().fields}
            retry_cols = [c for c in table.column_names if c in iceberg_cols]
            log_event("gold_iceberg_schema_mismatch", level="warning", error=str(exc), retry_cols=retry_cols)
            tbl.append(table.select(retry_cols))
            break
    else:
        if last_err is not None:
            raise last_err
    log_event("gold_iceberg_appended", table=identifier, records=table.num_rows)


def _write_gold_iceberg_chunked(
    table: pa.Table,
    *,
    namespace: str = "imdb",
    table_name: str = "gold",
    chunk_size: int = 25,
) -> None:
    """Write gold table to Iceberg in chunks to reduce peak MinIO disk usage.

    MinIO can hit ACCESS_DENIED during multipart upload when disk is constrained.
    Small chunks (25 rows ~200KB) reduce peak staging and improve retry success.
    """
    total = table.num_rows
    print(
        f"[embedding] Writing {total} rows to Iceberg {namespace}.{table_name} in chunks of {chunk_size}...", flush=True
    )
    written = 0
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = table.slice(start, end - start)
        _write_gold_iceberg(chunk, namespace=namespace, table_name=table_name)
        written += chunk.num_rows
        chunk_num = (written + chunk_size - 1) // chunk_size
        if chunk_num == 1 or chunk_num % 10 == 0 or written == total:
            print(f"[embedding] Iceberg: {written}/{total} rows written", flush=True)
    print("[embedding] Done.", flush=True)


def _load_bert_for_embedding(
    model_name: str, bert_path: Path | None, device: str
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load BERT (or fine-tuned BERT base) for encoding. Returns (model, tokenizer)."""
    if bert_path is not None and bert_path.is_dir() and (bert_path / "config.json").exists():
        # Fine-tuned model: load BertForSequenceClassification, use .bert for embeddings
        tokenizer = AutoTokenizer.from_pretrained(str(bert_path))
        full_model = BertForSequenceClassification.from_pretrained(str(bert_path))
        model = full_model.bert.to(device)
        model.eval()
        return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def run_embedding_job(
    silver_prefix: str | None = None,
    gold_prefix: str | None = None,
    model_name: str = DEFAULT_MODEL,
    bert_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    iceberg: bool = False,
    iceberg_namespace: str = "imdb",
    iceberg_table: str = "gold",
) -> int:
    """Read silver JSONL, encode with BERT, write gold Parquet or Iceberg.

    When bert_path is set to a fine-tuned model dir, uses its BERT base for embeddings.

    Returns
    -------
    Number of records written to gold.
    """
    silver_prefix = silver_prefix or f"{config.SILVER_PREFIX}imdb/"
    gold_prefix = gold_prefix or f"{config.GOLD_PREFIX}imdb/"

    ensure_bucket_exists()

    records = load_silver_records(silver_prefix)
    if not records:
        log_event("embedding_job_no_silver", level="warning", prefix=silver_prefix)
        return 0

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    load_from = str(bert_path) if bert_path else model_name
    log_event("embedding_job_loading_bert", model=load_from, device=device)
    print(f"[embedding] Loading BERT ({load_from}) on {device}...", flush=True)
    model, tokenizer = _load_bert_for_embedding(model_name, bert_path, device)
    print(f"[embedding] BERT loaded. Encoding {len(records)} records in batches of {batch_size}...", flush=True)
    log_event("embedding_job_bert_loaded", model=load_from, records=len(records), batch_size=batch_size)

    ids = []
    texts = []
    labels = []
    splits: list[str] = []
    request_ids: list[str | None] = []
    embeddings_list = []

    num_batches = (len(records) + batch_size - 1) // batch_size
    for i in range(0, len(records), batch_size):
        batch_num = i // batch_size + 1
        if batch_num % 10 == 0 or batch_num == num_batches:
            done = min(i + batch_size, len(records))
            print(f"[embedding] Batch {batch_num}/{num_batches} ({done}/{len(records)} records)", flush=True)
            log_event(
                "embedding_job_progress",
                batch=batch_num,
                total=num_batches,
                records_so_far=done,
                total_records=len(records),
            )
        batch = records[i : i + batch_size]
        batch_texts = [r.get("text", "") for r in batch]
        enc = encode_batch(model, tokenizer, batch_texts, torch.device(device))
        for j, r in enumerate(batch):
            ids.append(str(r.get("id", "")))
            texts.append(r.get("text", ""))
            labels.append(int(r.get("label", 0)))
            splits.append(str(r.get("split", "unknown")))
            request_ids.append(r.get("request_id"))
            embeddings_list.append([float(x) for x in enc[j].cpu().numpy().astype("float32").tolist()])

    from src.utils.schema import gold_embedding_arrow_schema

    table = pa.table(
        {
            "id": ids,
            "text": texts,
            "label": labels,
            "embedding": embeddings_list,
            "split": splits,
            "request_id": request_ids,
        },
        schema=gold_embedding_arrow_schema(),
    )

    if iceberg:
        _write_gold_iceberg_chunked(
            table,
            namespace=iceberg_namespace,
            table_name=iceberg_table,
            chunk_size=25,
        )
    else:
        _write_gold_parquet(table, gold_prefix)
    return len(records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode silver IMDb reviews with BERT and write gold Parquet.")
    parser.add_argument(
        "--silver-prefix",
        type=str,
        default=None,
        help="Silver path prefix (default: config.SILVER_PREFIX + 'imdb/').",
    )
    parser.add_argument(
        "--gold-prefix",
        type=str,
        default=None,
        help="Gold path prefix (default: config.GOLD_PREFIX + 'imdb/').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"BERT model name when not using --bert-path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--bert-path",
        type=str,
        default=None,
        help="Path to fine-tuned BERT dir (e.g. models/bert_sentiment_imdb). Overrides --model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for encoding (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--iceberg",
        action="store_true",
        help="Write to Iceberg table instead of Parquet.",
    )
    parser.add_argument(
        "--iceberg-namespace",
        type=str,
        default="imdb",
        help="Iceberg namespace to use when --iceberg is set (default: 'imdb').",
    )
    parser.add_argument(
        "--iceberg-table",
        type=str,
        default="gold",
        help="Iceberg table name to use when --iceberg is set (default: 'gold').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bert_path = None
    if args.bert_path:
        p = Path(args.bert_path)
        if not p.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            p = project_root / args.bert_path
        bert_path = p.resolve()
    total = run_embedding_job(
        silver_prefix=args.silver_prefix,
        gold_prefix=args.gold_prefix,
        model_name=args.model,
        bert_path=bert_path,
        batch_size=args.batch_size,
        device=args.device,
        iceberg=args.iceberg,
        iceberg_namespace=args.iceberg_namespace,
        iceberg_table=args.iceberg_table,
    )
    log_event("embedding_job_complete", total_records=total)


if __name__ == "__main__":
    main()
