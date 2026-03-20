"""Ingestion producer for streaming IMDb reviews into Kafka.

This module can be used as a CLI script. It will download the IMDb
Movie Reviews (binary sentiment) dataset at runtime, normalise each
record into a simple schema, and stream the reviews to Kafka.
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
import time
from collections.abc import Iterable

from datasets import load_dataset

from src import config
from src.utils.kafka_client import create_producer
from src.utils.schema import ImdbBronzeReview
from src.utils.structured_logging import log_event


def load_imdb_reviews(
    split: str | None = None,
    limit: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterable[ImdbBronzeReview]:
    """Load and normalise IMDb reviews from the datasets library.

    Parameters
    ----------
    split:
        Dataset split to load (e.g. ``\"train\"``, ``\"test\"``). If ``None``,
        uses :data:`config.IMDB_DEFAULT_SPLIT`.
    limit:
        Optional maximum number of records to yield.
    shuffle:
        If True, shuffle the dataset before streaming so positive/negative
        reviews are mixed (default: True).
    seed:
        Random seed for shuffling (default: 42).
    """

    split = split or config.IMDB_DEFAULT_SPLIT

    ds = load_dataset("imdb", split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    max_records = limit or config.IMDB_MAX_RECORDS or len(ds)

    for idx, row in enumerate(ds):
        if idx >= max_records:
            break
        yield ImdbBronzeReview.from_raw_imdb(row, idx, split=split)


def stream_imdb_reviews(
    mode: str = "batch",
    sleep_seconds: float = 1.0,
    split: str | None = None,
    limit: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """Stream IMDb reviews into Kafka.

    Parameters
    ----------
    mode:
        ``\"batch\"`` to send all available reviews as fast as possible,
        ``\"realtime\"`` to sleep ``sleep_seconds`` between messages.
    sleep_seconds:
        Delay between messages when ``mode=\"realtime\"``.
    split:
        Dataset split to use.
    limit:
        Optional maximum number of reviews to send.
    """

    producer = create_producer()
    topic = config.KAFKA_IMDB_TOPIC

    log_event("producer_started", topic=topic, servers=config.KAFKA_BOOTSTRAP_SERVERS, mode=mode)

    count = 0
    for review in load_imdb_reviews(split=split, limit=limit, shuffle=shuffle, seed=seed):
        producer.send(topic, review.to_message())
        count += 1

        if mode == "realtime":
            time.sleep(sleep_seconds)

    producer.flush()
    producer.close()
    log_event("producer_finished", topic=topic, count=count)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream IMDb Movie Reviews into Kafka.")
    parser.add_argument(
        "--mode",
        choices=["batch", "realtime"],
        default="realtime",
        help="Streaming mode: 'batch' sends as fast as possible; 'realtime' sleeps between messages.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Seconds to sleep between messages in realtime mode.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="IMDb dataset split to use (e.g. 'train', 'test'). Defaults to IMDB_DEFAULT_SPLIT from config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of reviews to send.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling; stream in dataset order (negatives first).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    stream_imdb_reviews(
        mode=args.mode,
        sleep_seconds=args.sleep_seconds,
        split=args.split,
        limit=args.limit,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
