#!/usr/bin/env python3
"""Check if IMDb data is already prepopulated (gold tables + trained model).

Exit 0 if prepopulated (skip full pipeline), exit 1 if not.
Used by prepopulate_imdb to avoid re-running when data already exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from pyiceberg.exceptions import NoSuchTableError

from src.utils.iceberg_catalog import get_iceberg_catalog


def is_prepopulated() -> bool:
    """Return True if gold tables and model exist with data."""
    logreg_path = Path("models/sentiment_logreg.joblib")
    finetuned_path = Path("models/bert_sentiment_imdb/config.json")
    if not logreg_path.exists() and not finetuned_path.exists():
        return False

    try:
        catalog = get_iceberg_catalog()
        for identifier in ("imdb.gold_train", "imdb.gold_test"):
            try:
                tbl = catalog.load_table(identifier)
                df = tbl.scan().to_pandas()
                if len(df) == 0:
                    return False
            except NoSuchTableError:
                return False
        return True
    except (OSError, Exception):
        return False


def main() -> int:
    if is_prepopulated():
        print("Already prepopulated (gold tables + model present). Skipping.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
