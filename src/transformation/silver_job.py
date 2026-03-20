"""Silver layer transformation job.

Reads bronze JSONL files from MinIO, applies text cleaning, and writes
cleaned records to the silver layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import argparse
import json
import time
from uuid import uuid4

from src import config
from src.transformation.cleaning import clean_text
from src.utils.s3_client import ensure_bucket_exists, get_object_body, list_objects, upload_bytes
from src.utils.schema import ImdbSilverReview
from src.utils.structured_logging import log_event


def deduplicate_by_text_label(
    records: list[ImdbSilverReview],
    seen: set[tuple[str, int, str]] | None = None,
) -> list[ImdbSilverReview]:
    """Keep first occurrence of each (text, label) pair.

    If ``seen`` is provided, uses and updates it for cross-batch deduplication.
    """
    if seen is None:
        seen = set()
    result: list[ImdbSilverReview] = []
    for r in records:
        key = (r.text, int(r.label), r.split)
        if key in seen:
            continue
        seen.add(key)
        result.append(r)
    return result


def run_silver_job(
    bronze_prefix: str | None = None,
    silver_prefix: str | None = None,
    dedup: bool = True,
) -> int:
    """Read bronze JSONL files, clean text, deduplicate, write to silver.

    Deduplication (when enabled) keeps the first occurrence of each (text, label)
    pair to avoid redundant records for downstream embedding and training.

    Returns
    -------
    Total number of records written to silver.
    """
    bronze_prefix = bronze_prefix or f"{config.BRONZE_PREFIX}imdb/"
    silver_prefix = silver_prefix or f"{config.SILVER_PREFIX}imdb/"

    ensure_bucket_exists()

    objects = list_objects(bronze_prefix)
    if not objects:
        log_event("silver_job_no_bronze", level="warning", prefix=bronze_prefix)
        return 0

    seen: set[tuple[str, int, str]] = set()
    total = 0
    total_skipped = 0

    for obj in objects:
        key = obj["Key"]
        if not key.endswith(".jsonl"):
            continue

        body = get_object_body(key)
        lines = [ln.strip() for ln in body.decode("utf-8").strip().split("\n") if ln.strip()]

        silver_list: list[ImdbSilverReview] = []
        for line in lines:
            try:
                rec = json.loads(line)
                label_raw = rec.get("label", 0)
                if label_raw not in (0, 1):
                    raise ValueError(f"Invalid label: {label_raw}")
                raw_text = str(rec.get("text", ""))
                cleaned = clean_text(raw_text)
                silver = ImdbSilverReview(
                    id=str(rec.get("id", "")),
                    text=cleaned,
                    label=label_raw,
                    split=rec.get("split", "unknown"),
                    request_id=rec.get("request_id"),
                )
                silver_list.append(silver)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                log_event("silver_invalid_record", level="warning", key=key, error=str(e))
                continue

        if dedup:
            before = len(silver_list)
            silver_list = deduplicate_by_text_label(silver_list, seen=seen)
            total_skipped += before - len(silver_list)

        if not silver_list:
            continue

        silver_records = [json.dumps(s.to_dict(), ensure_ascii=False) for s in silver_list]
        out_key = f"{silver_prefix}imdb_silver_{int(time.time())}_{uuid4().hex}.jsonl"
        data = ("\n".join(silver_records) + "\n").encode("utf-8")
        upload_bytes(out_key, data)
        total += len(silver_list)
        log_event("silver_batch_written", key=out_key, records=len(silver_list))

    if dedup and total_skipped > 0:
        log_event("silver_dedup_summary", skipped=total_skipped)
    return total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform bronze IMDb reviews to silver (cleaned) layer.")
    parser.add_argument(
        "--bronze-prefix",
        type=str,
        default=None,
        help="Bronze path prefix (default: config.BRONZE_PREFIX + 'imdb/').",
    )
    parser.add_argument(
        "--silver-prefix",
        type=str,
        default=None,
        help="Silver path prefix (default: config.SILVER_PREFIX + 'imdb/').",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication on (text, label).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    total = run_silver_job(
        bronze_prefix=args.bronze_prefix,
        silver_prefix=args.silver_prefix,
        dedup=not args.no_dedup,
    )
    log_event("silver_job_complete", total_records=total)


if __name__ == "__main__":
    main()
