"""Minimal structured logging helper for the demo.

Prints JSON to stdout so container logs are easier to parse.
All modules should use ``log_event`` instead of bare ``print()`` for
operational messages (errors, progress, lifecycle events).
"""

from __future__ import annotations

import json
import time
from typing import Any


def log_event(event: str, *, level: str = "info", **fields: Any) -> None:
    """Emit a structured JSON log line to stdout.

    Parameters
    ----------
    event:
        Short snake_case event name (e.g. ``"silver_job_started"``).
    level:
        Log severity -- ``"debug"``, ``"info"``, ``"warning"``, ``"error"``.
    **fields:
        Arbitrary key-value pairs included in the JSON payload.
    """
    payload = {
        "ts_ms": int(time.time() * 1000),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


__all__ = ["log_event"]
