"""Client for generating synthetic IMDb-style reviews with local Ollama.

This client calls Ollama's OpenAI-compatible endpoint and publishes generated
reviews to Kafka using the same message schema as the ingestion layer.
"""

from __future__ import annotations

import argparse
import time
import uuid

import requests

from src import config
from src.utils.kafka_client import create_producer
from src.utils.schema import ImdbBronzeReview
from src.utils.structured_logging import log_event


def openai_chat_completion(api_base: str, model: str, prompt: str, max_tokens: int) -> str:
    """Call an OpenAI-compatible /v1/chat/completions endpoint."""
    base = api_base.rstrip("/")
    # Ollama uses base http://localhost:11434/v1; avoid double /v1.
    url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.9,
    }
    last_exc: Exception | None = None
    for _attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2)
    raise RuntimeError(f"LLM request failed after retries: {last_exc}")


def build_prompt(sentiment: str) -> str:
    """Prompt that asks for an IMDb-like review with explicit sentiment."""
    if sentiment == "positive":
        return (
            "Write an IMDb-style movie review that is clearly POSITIVE. "
            "Do not include the words 'positive' or 'negative' in the text. "
            "Aim for 2-4 paragraphs."
        )
    if sentiment == "negative":
        return (
            "Write an IMDb-style movie review that is clearly NEGATIVE. "
            "Do not include the words 'positive' or 'negative' in the text. "
            "Aim for 2-4 paragraphs."
        )
    raise ValueError("sentiment must be 'positive' or 'negative'")


def generate_and_stream(
    *,
    sentiment: str,
    count: int,
    split: str,
    kafka_topic: str,
    api_base: str,
    model: str,
    max_tokens: int,
) -> None:
    producer = create_producer()

    label = 1 if sentiment == "positive" else 0
    log_event("llm_generate_started", sentiment=sentiment, count=count, topic=kafka_topic)

    for _ in range(count):
        prompt = build_prompt(sentiment)
        text = openai_chat_completion(api_base=api_base, model=model, prompt=prompt, max_tokens=max_tokens)

        rid = str(uuid.uuid4())
        record = ImdbBronzeReview(id=rid, text=text, label=label, split=split, request_id=rid)
        producer.send(kafka_topic, record.to_message())

    producer.flush()
    producer.close()
    log_event("llm_generate_finished", sentiment=sentiment, count=count)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic reviews with local LLM and stream to Kafka.")
    parser.add_argument("--sentiment", type=str, choices=["positive", "negative"], required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--split", type=str, default="train", help="Split to write into the bronze/silver/gold pipeline."
    )
    parser.add_argument("--kafka-topic", type=str, default=None, help="Kafka topic (default: config.KAFKA_IMDB_TOPIC).")
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:11434/v1",
        help="Ollama API base (default: http://localhost:11434/v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model name (e.g. llama3.2, mistral).",
    )
    parser.add_argument("--max-tokens", type=int, default=350)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    topic = args.kafka_topic or config.KAFKA_IMDB_TOPIC
    generate_and_stream(
        sentiment=args.sentiment,
        count=args.count,
        split=args.split,
        kafka_topic=topic,
        api_base=args.api_base,
        model=args.model,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
