"""Send an inference request to Kafka for async processing (demo)."""

from __future__ import annotations

import argparse
import uuid

from src import config
from src.utils.kafka_client import create_producer
from src.utils.structured_logging import log_event


def send_request(text: str, request_id: str, split: str = "inference", label: int = 0) -> None:
    producer = create_producer()
    topic = config.KAFKA_INFERENCE_TOPIC
    payload = {"id": request_id, "text": text, "label": int(label), "split": split, "request_id": request_id}
    producer.send(topic, payload)
    producer.flush()
    producer.close()
    log_event("inference_request_sent", topic=topic, request_id=request_id)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send async inference request to Kafka (demo).")
    parser.add_argument("--text", type=str, required=True, help="Review text to classify.")
    parser.add_argument("--request-id", type=str, default=None, help="Request id (default: random UUID).")
    parser.add_argument("--split", type=str, default="inference", help="Message split (default: inference).")
    parser.add_argument("--label", type=int, default=0, help="Dummy label; predictions are stored separately.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rid = args.request_id or str(uuid.uuid4())
    send_request(text=args.text, request_id=rid, split=args.split, label=args.label)


if __name__ == "__main__":
    main()
