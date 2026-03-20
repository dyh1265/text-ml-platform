"""Prometheus metrics for monitoring prediction and inference services.

Metrics are defined here so they can be imported by predict_service and
inference_worker. The /metrics endpoint is exposed by the predict service
for scraping by Prometheus (or manual inspection).
"""

from __future__ import annotations


class _NoOpMetric:
    """No-op metric when prometheus_client is not installed."""

    def labels(self, **kwargs):
        return self

    def inc(self, amount=1):
        pass

    def observe(self, amount):
        pass


try:
    from prometheus_client import Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True

    def _counter(*args, **kwargs):
        return Counter(*args, **kwargs)

    def _histogram(*args, **kwargs):
        return Histogram(*args, **kwargs)
except ImportError:
    PROMETHEUS_AVAILABLE = False
    generate_latest = None  # type: ignore[misc, assignment]
    _noop = _NoOpMetric()

    def _counter(*args, **kwargs):
        return _noop

    def _histogram(*args, **kwargs):
        return _noop


# Prediction API metrics
PREDICT_REQUESTS_TOTAL = _counter(
    "predict_requests_total",
    "Total prediction requests received",
    ["split"],
)
PREDICT_LATENCY_SECONDS = _histogram(
    "predict_latency_seconds",
    "Prediction request latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
PREDICT_LABEL_TOTAL = _counter(
    "predict_label_total",
    "Predictions by label (0=negative, 1=positive)",
    ["label"],
)

# Inference worker metrics
INFERENCE_CONSUMED_TOTAL = _counter(
    "inference_consumed_total",
    "Total messages consumed from Kafka (inference split)",
)
INFERENCE_SUCCESS_TOTAL = _counter(
    "inference_success_total",
    "Total inference requests processed successfully",
)
INFERENCE_FAILURE_TOTAL = _counter(
    "inference_failure_total",
    "Total inference requests that failed (sent to DLQ)",
)
INFERENCE_PROCESSING_SECONDS = _histogram(
    "inference_processing_seconds",
    "Time to process a single inference request",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)


def get_metrics_body() -> bytes:
    """Return Prometheus text format for all registered metrics."""
    if not PROMETHEUS_AVAILABLE or generate_latest is None:
        return b"# Prometheus client not installed\n"
    return generate_latest()


__all__ = [
    "INFERENCE_CONSUMED_TOTAL",
    "INFERENCE_FAILURE_TOTAL",
    "INFERENCE_PROCESSING_SECONDS",
    "INFERENCE_SUCCESS_TOTAL",
    "PREDICT_LABEL_TOTAL",
    "PREDICT_LATENCY_SECONDS",
    "PREDICT_REQUESTS_TOTAL",
    "PROMETHEUS_AVAILABLE",
    "get_metrics_body",
]
