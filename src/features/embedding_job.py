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
from typing import Optional

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import argparse

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

from src import config
from src.utils.s3_client import ensure_bucket_exists, get_object_body, list_objects, upload_bytes

DEFAULT_MODEL = "bert-base-uncased"
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
    print(f"Wrote {table.num_rows} records to s3://{config.S3_DATA_BUCKET}/{out_key}")


def _write_gold_iceberg(table: pa.Table, namespace: str = "imdb", table_name: str = "gold") -> None:
    """Write gold table to Iceberg in MinIO."""
    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        FloatType,
        ListType,
        LongType,
        NestedField,
        StringType,
    )

    from src.utils.iceberg_catalog import get_iceberg_catalog

    catalog = get_iceberg_catalog()
    catalog.create_namespace_if_not_exists(namespace)

    schema = Schema(
        NestedField(1, "id", StringType(), required=True),
        NestedField(2, "text", StringType(), required=True),
        NestedField(3, "label", LongType(), required=True),
        NestedField(4, "embedding", ListType(element_id=5, element_type=FloatType(), element_required=False), required=True),
    )

    identifier = f"{namespace}.{table_name}"
    try:
        tbl = catalog.load_table(identifier)
    except Exception:
        tbl = catalog.create_table(identifier, schema=schema)
    tbl.append(table)
    print(f"Appended {table.num_rows} records to Iceberg table {identifier}")


def run_embedding_job(
    silver_prefix: Optional[str] = None,
    gold_prefix: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    iceberg: bool = False,
    iceberg_namespace: str = "imdb",
    iceberg_table: str = "gold",
) -> int:
    """Read silver JSONL, encode with BERT, write gold Parquet or Iceberg.

    Returns
    -------
    Number of records written to gold.
    """
    silver_prefix = silver_prefix or f"{config.SILVER_PREFIX}imdb/"
    gold_prefix = gold_prefix or f"{config.GOLD_PREFIX}imdb/"

    ensure_bucket_exists()

    records = load_silver_records(silver_prefix)
    if not records:
        print(f"No silver records found under {silver_prefix}")
        return 0

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    ids = []
    texts = []
    labels = []
    embeddings_list = []

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_texts = [r.get("text", "") for r in batch]
        enc = encode_batch(model, tokenizer, batch_texts, torch.device(device))
        for j, r in enumerate(batch):
            ids.append(str(r.get("id", "")))
            texts.append(r.get("text", ""))
            labels.append(int(r.get("label", 0)))
            embeddings_list.append([float(x) for x in enc[j].cpu().numpy().astype("float32").tolist()])

    # Iceberg requires non-nullable cols and list<float> (float32)
    schema = pa.schema([
        pa.field("id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("label", pa.int64(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    ])
    table = pa.table(
        {"id": ids, "text": texts, "label": labels, "embedding": embeddings_list},
        schema=schema,
    )

    if iceberg:
        _write_gold_iceberg(table, namespace=iceberg_namespace, table_name=iceberg_table)
    else:
        _write_gold_parquet(table, gold_prefix)
    return len(records)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode silver IMDb reviews with BERT and write gold Parquet."
    )
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
        help=f"BERT model name (default: {DEFAULT_MODEL}).",
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


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    total = run_embedding_job(
        silver_prefix=args.silver_prefix,
        gold_prefix=args.gold_prefix,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        iceberg=args.iceberg,
        iceberg_namespace=args.iceberg_namespace,
        iceberg_table=args.iceberg_table,
    )
    print(f"Embedding job complete. Wrote {total} records.")
