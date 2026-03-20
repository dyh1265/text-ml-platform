"""Tests for centralized config."""

from src import config


def test_config_exports_expected_vars():
    """Config module exports all expected configuration keys."""
    expected = [
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_IMDB_TOPIC",
        "KAFKA_INFERENCE_TOPIC",
        "KAFKA_IMDB_DLQ_TOPIC",
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
        "S3_DATA_BUCKET",
        "BRONZE_PREFIX",
        "SILVER_PREFIX",
        "GOLD_PREFIX",
        "FEATURES_PREFIX",
        "ICEBERG_CATALOG_DB",
        "PREDICT_API_URL",
        "GOLD_ICEBERG_IDENTIFIER",
        "PCA_MAX_POINTS",
        "OLLAMA_API_BASE",
        "DEFAULT_MODEL_PATH",
        "ASYNC_PREDICTION_TIMEOUT_S",
    ]
    for name in expected:
        assert hasattr(config, name), f"config missing {name}"


def test_kafka_bootstrap_servers_is_list():
    assert isinstance(config.KAFKA_BOOTSTRAP_SERVERS, list)
    assert len(config.KAFKA_BOOTSTRAP_SERVERS) >= 1


def test_iceberg_catalog_db_is_path_string():
    assert isinstance(config.ICEBERG_CATALOG_DB, str)
    assert "catalog.db" in config.ICEBERG_CATALOG_DB or "iceberg" in config.ICEBERG_CATALOG_DB.lower()


def test_pca_max_points_is_int():
    assert isinstance(config.PCA_MAX_POINTS, int)
    assert config.PCA_MAX_POINTS > 0
