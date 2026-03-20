"""Pytest configuration: ensure project root is on sys.path for 'src' imports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    """Ignore httpx TestClient deprecation warning (app shortcut -> WSGITransport)."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The 'app' shortcut is now deprecated:DeprecationWarning",
    )
