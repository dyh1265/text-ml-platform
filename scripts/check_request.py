import sys
from pathlib import Path
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
rid = sys.argv[1] if len(sys.argv) > 1 else "80baf972-4ca9-41cc-8562-0af8cce69144"
from pyiceberg.expressions import EqualTo
from src.utils.iceberg_catalog import get_iceberg_catalog
cat = get_iceberg_catalog()
for name in ("imdb.gold_inference", "imdb.predictions"):
    try:
        tbl = cat.load_table(name)
        df = tbl.scan(row_filter=EqualTo("request_id", rid)).to_pandas()
        print(f"{name}: {len(df)} row(s)")
        if len(df) > 0:
            print(df.to_string())
        else:
            print("  (not found)")
    except Exception as e:
        print(f"{name}: ERROR - {e}")
    print()
