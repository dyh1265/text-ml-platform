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
from typing import List, Optional
from uuid import uuid4

from kafka import KafkaConsumer

from src import config
from src.utils.schema import ImdbBronzeReview
from src.utils.s3_client import ensure_bucket_exists, upload_bytes


def _flush_batch(lines: List[str], split: str) -> None:
    """Upload a batch of JSONL records to the bronze layer."""
    if not lines:
        return

    ensure_bucket_exists()

    timestamp = int(time.time())
    key = (
        f"{config.BRONZE_PREFIX}imdb/{split}/"
        f"imdb_bronze_{timestamp}_{uuid4().hex}.jsonl"
    )
    data = ("\n".join(lines) + "\n").encode("utf-8")
    upload_bytes(key, data)
    print(f"Wrote {len(lines)} records to s3://{config.S3_DATA_BUCKET}/{key}")


def consume_to_bronze(batch_size: int = 1000, split: str = "train") -> None:
    """Consume IMDb reviews from Kafka and write them to bronze."""
    split = split or config.IMDB_DEFAULT_SPLIT
    consumer = KafkaConsumer(
        config.KAFKA_IMDB_TOPIC,
        bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        group_id="imdb-bronze-consumer",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    print(
        f"Consuming IMDb reviews from topic '{config.KAFKA_IMDB_TOPIC}' "
        f"on {config.KAFKA_BOOTSTRAP_SERVERS}..."
        f"from split '{split}'"
    )

    buffer: List[str] = []

    try:
        for msg in consumer:
            payload = msg.value
            try:
                record = ImdbBronzeReview(
                    id=str(payload.get("id")),
                    text=str(payload.get("text", "")),
                    label=int(payload.get("label")),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping invalid message: {payload!r} ({exc})")
                continue

            buffer.append(json.dumps(record.to_message(), ensure_ascii=False))

            if len(buffer) >= batch_size:
                _flush_batch(buffer, split)
                buffer.clear()
    except KeyboardInterrupt:
        print("Stopping consumer...")
    finally:
        if buffer:
            _flush_batch(buffer, split)
        consumer.close()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    consume_to_bronze(batch_size=args.batch_size, split=args.split)


if __name__ == "__main__":
    main()