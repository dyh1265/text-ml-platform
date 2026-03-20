#!/usr/bin/env python3
"""Quick diagnostic for imdb.predictions table.

Run from project root (or with PYTHONPATH set):
  python scripts/check_predictions.py

Useful when async UI shows bronze/silver/gold complete but prediction row never appears.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.utils.iceberg_catalog import get_iceberg_catalog

def main():
    catalog = get_iceberg_catalog()
    try:
        tbl = catalog.load_table("imdb.predictions")
        df = tbl.scan().to_pandas()
        print(f"imdb.predictions: {len(df)} row(s)")
        if len(df) > 0:
            print(df.tail(5).to_string())
        else:
            print("  (table exists but is empty)")
    except Exception as e:
        print(f"imdb.predictions: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
