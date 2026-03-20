"""Bronze layer consumer for IMDb reviews.

This script consumes normalised IMDb review messages from Kafka and
writes them into the bronze layer in object storage (MinIO/S3) as
JSON Lines files.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as script: add project root to path
if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import argparse
import json
import time
from uuid import uuid4

from kafka import KafkaConsumer

from src import config
from src.utils.s3_client import ensure_bucket_exists, upload_bytes
from src.utils.schema import ImdbBronzeReview
from src.utils.structured_logging import log_event


def _flush_batch(lines: list[str], split: str) -> None:
    """Upload a batch of JSONL records to the bronze layer."""
    if not lines:
        return

    ensure_bucket_exists()

    timestamp = int(time.time())
    key = f"{config.BRONZE_PREFIX}imdb/{split}/imdb_bronze_{timestamp}_{uuid4().hex}.jsonl"
    data = ("\n".join(lines) + "\n").encode("utf-8")
    upload_bytes(key, data)
    log_event("bronze_batch_written", key=key, records=len(lines))


def consume_to_bronze(
    batch_size: int = 1000,
    split: str = "train",
    limit: int | None = None,
    group_id: str | None = None,
) -> None:
    """Consume IMDb reviews from Kafka and write them to bronze.

    When limit is set, consumes up to that many messages and exits (for batch/prepopulate).
    Uses a unique group_id when limit is set so batch runs always read from earliest,
    avoiding committed offsets from previous runs.
    """
    split = split or config.IMDB_DEFAULT_SPLIT
    effective_group = group_id or (
        f"imdb-bronze-consumer-batch-{int(time.time())}" if limit is not None else "imdb-bronze-consumer"
    )
    consumer = KafkaConsumer(
        config.KAFKA_IMDB_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        group_id=effective_group,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    log_event("bronze_consumer_started", topic=config.KAFKA_IMDB_TOPIC, split=split, limit=limit)

    buffer_by_split: dict[str, list[str]] = {}
    consumed = 0

    try:
        for msg in consumer:
            payload = msg.value
            try:
                request_id = payload.get("request_id")
                record_split = str(payload.get("split") or split)
                record = ImdbBronzeReview(
                    id=str(payload.get("id")),
                    text=str(payload.get("text", "")),
                    label=int(payload.get("label") or 0),
                    split=record_split,
                    request_id=request_id,
                )
            except (ValueError, KeyError, TypeError) as exc:
                log_event("bronze_invalid_message", level="warning", error=str(exc))
                continue

            buffer_by_split.setdefault(record.split, []).append(json.dumps(record.to_message(), ensure_ascii=False))
            consumed += 1

            if len(buffer_by_split[record.split]) >= batch_size:
                _flush_batch(buffer_by_split[record.split], record.split)
                buffer_by_split[record.split].clear()

            if limit is not None and consumed >= limit:
                log_event("bronze_consumer_limit_reached", consumed=consumed, limit=limit)
                break
    except KeyboardInterrupt:
        log_event("bronze_consumer_stopping")
    finally:
        for buffered_split, buffered_lines in buffer_by_split.items():
            if buffered_lines:
                _flush_batch(buffered_lines, buffered_split)
        consumer.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consume IMDb reviews from Kafka into the bronze layer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records to buffer before writing a JSONL object.",
    )
    parser.add_argument("--split", type=str, default="train", help="Split to consume from.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Consume up to N messages and exit (for batch/prepopulate). Default: run until interrupt.",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=None,
        help="Kafka consumer group ID. When --limit is set, defaults to a unique ID so batch runs read from earliest.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    consume_to_bronze(batch_size=args.batch_size, split=args.split, limit=args.limit, group_id=args.group_id)


if __name__ == "__main__":
    main()
