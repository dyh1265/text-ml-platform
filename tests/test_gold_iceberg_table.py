import pytest

from src import config
from src.utils.iceberg_catalog import get_iceberg_catalog


@pytest.mark.integration
def test_gold_iceberg_table_loads_and_has_snapshots():
    """
    Sanity check mirroring the Streamlit sidebar logic:

    - Load the Iceberg catalog configured for MinIO.
    - Load the gold table (default: imdb.gold_train).
    - Assert that we can access metadata and that at least one snapshot exists.

    This verifies that:
    - MinIO/S3 credentials are wired correctly for Iceberg.
    - The embedding job + prepopulation have created the expected table.
    """
    identifier = config.GOLD_ICEBERG_IDENTIFIER
    namespace, table_name = identifier.split(".", 1)

    catalog = get_iceberg_catalog()
    table = catalog.load_table(f"{namespace}.{table_name}")

    snapshots = getattr(table.metadata, "snapshots", [])
    # We expect at least one snapshot after prepopulation has run.
    assert snapshots, f"No snapshots found for Iceberg table {identifier!r}"
