"""Tests for train_classifier."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.training.train_classifier import _apply_split_filter, train_and_save


def test_apply_split_filter_no_split_col():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = _apply_split_filter(df, split_col="split", split_value="train")
    assert len(result) == 2


def test_apply_split_filter_none_value():
    df = pd.DataFrame({"split": ["train"], "x": [1]})
    result = _apply_split_filter(df, split_col="split", split_value=None)
    assert len(result) == 1


def test_apply_split_filter_filters():
    df = pd.DataFrame({"split": ["train", "test", "train"], "x": [1, 2, 3]})
    result = _apply_split_filter(df, split_col="split", split_value="train")
    assert len(result) == 2


def test_train_and_save():
    mock_df = pd.DataFrame({
        "embedding": [[0.1]*768, [0.2]*768, [0.9]*768, [0.8]*768],
        "label": [0, 0, 1, 1],
        "split": ["train", "train", "train", "train"],
    })
    mock_scan = MagicMock()
    mock_scan.to_pandas.return_value = mock_df
    mock_table = MagicMock()
    mock_table.scan.return_value = mock_scan
    mock_catalog = MagicMock()
    mock_catalog.load_table.return_value = mock_table

    with (patch("src.training.train_classifier.get_iceberg_catalog", return_value=mock_catalog),
          patch("src.training.train_classifier.joblib.dump")):
        out_path = Path("models/test_sentiment.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        train_and_save(
            iceberg_identifier="imdb.gold_train",
            train_split="train",
            test_split=None,
            test_iceberg_identifier=None,
            embedding_col="embedding",
            label_col="label",
            split_col="split",
            max_records=None,
            random_seed=42,
            model_out=out_path,
        )
        mock_catalog.load_table.assert_called()
        if out_path.exists():
            out_path.unlink()
