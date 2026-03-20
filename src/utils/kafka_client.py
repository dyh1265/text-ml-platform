"""Shared Kafka producer factory with retry logic.

All modules that need a KafkaProducer should import ``create_producer``
from here instead of duplicating the retry loop.
"""

from __future__ import annotations

import json
import time

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from src import config
from src.utils.structured_logging import log_event

_MAX_RETRIES = 10
_RETRY_DELAY_S = 2


def create_producer(
    *,
    bootstrap_servers: list[str] | None = None,
    retries: int = _MAX_RETRIES,
) -> KafkaProducer:
    """Create a KafkaProducer with automatic retry on broker unavailability.

    Parameters
    ----------
    bootstrap_servers:
        Override for Kafka brokers; defaults to ``config.KAFKA_BOOTSTRAP_SERVERS``.
    retries:
        Number of connection attempts before raising.
    """
    servers = bootstrap_servers or config.KAFKA_BOOTSTRAP_SERVERS
    last_error: Exception | None = None

    for _ in range(retries):
        try:
            return KafkaProducer(
                bootstrap_servers=servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except NoBrokersAvailable as exc:
            last_error = exc
            log_event("kafka_broker_retry", level="warning")
            time.sleep(_RETRY_DELAY_S)

    raise RuntimeError(f"Kafka broker not reachable at {servers}") from last_error


__all__ = ["create_producer"]
