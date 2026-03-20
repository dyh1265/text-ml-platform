"""Train a simple sentiment classifier on top of gold embeddings.

This is meant for the production-demo: a trained model can power the
prediction API used by the Streamlit UI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyiceberg.exceptions import NoSuchTableError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.utils.iceberg_catalog import get_iceberg_catalog
from src.utils.structured_logging import log_event


def _load_table_df(iceberg_identifier: str) -> pd.DataFrame:
    """Load an Iceberg table as a pandas DataFrame.
    If the requested table does not exist, tries imdb.gold as fallback.
    """
    catalog = get_iceberg_catalog()
    namespace, table_name = iceberg_identifier.split(".", 1)
    try:
        tbl = catalog.load_table(f"{namespace}.{table_name}")
    except NoSuchTableError:
        # Fallback: often imdb.gold exists from earlier runs without split-specific tables.
        if iceberg_identifier != "imdb.gold":
            log_event("train_table_fallback", level="warning", requested=iceberg_identifier, fallback="imdb.gold")
            try:
                tbl = catalog.load_table("imdb.gold")
            except NoSuchTableError:
                raise NoSuchTableError(
                    f"Table {iceberg_identifier} does not exist and imdb.gold was not found. "
                    "Run the embedding job first, e.g.:\n"
                    "  python -m src.features.embedding_job --silver-prefix silver/imdb/train/ "
                    "--iceberg --iceberg-table gold_train"
                ) from None
        else:
            raise
    return tbl.scan().to_pandas()


def _apply_split_filter(df: pd.DataFrame, split_col: str, split_value: str | None) -> pd.DataFrame:
    if split_value is None:
        return df
    if split_col not in df.columns:
        return df
    return df[df[split_col] == split_value]


def train_and_save(
    iceberg_identifier: str,
    train_split: str,
    test_split: str | None,
    test_iceberg_identifier: str | None,
    embedding_col: str,
    label_col: str,
    split_col: str,
    max_records: int | None,
    random_seed: int,
    model_out: Path,
) -> None:
    df = _load_table_df(iceberg_identifier)

    # Filter to training split (if the column exists).
    df_train = _apply_split_filter(df, split_col=split_col, split_value=train_split)
    if len(df_train) == 0 and split_col in df.columns:
        # Backwards-compat: if split column exists but doesn't contain the requested values,
        # fall back to training on all rows.
        log_event("train_split_fallback", level="warning", split=train_split)
        df_train = df
    if max_records is not None and len(df_train) > max_records:
        df_train = df_train.sample(n=max_records, random_state=random_seed)

    if len(df_train) == 0:
        raise ValueError(f"No training rows found for split={train_split!r} in {iceberg_identifier}")

    X_train = np.array(df_train[embedding_col].tolist(), dtype=np.float32)
    y_train = df_train[label_col].astype(int).to_numpy()

    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Training data has only one class (labels: {unique_labels.tolist()}). "
            "Logistic regression needs both positive and negative samples. "
            "Stream more data so that bronze/silver/gold include both label 0 and 1, or check that the producer sends mixed labels."
        )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    metrics = {"train_accuracy": float(accuracy_score(y_train, clf.predict(X_train)))}

    # Test set: from same table (filter by split) or from separate --test-iceberg-identifier table.
    df_test = None
    if test_split is not None:
        if test_iceberg_identifier is not None:
            df_test = _load_table_df(test_iceberg_identifier)
            df_test = _apply_split_filter(df_test, split_col=split_col, split_value=test_split)
            if len(df_test) == 0:
                df_test = _load_table_df(test_iceberg_identifier)  # use all rows if no split match
        else:
            df_test = _apply_split_filter(df, split_col=split_col, split_value=test_split)
    if df_test is not None and len(df_test) > 0:
        if max_records is not None and len(df_test) > max_records:
            df_test = df_test.sample(n=max_records, random_state=random_seed)
        X_test = np.array(df_test[embedding_col].tolist(), dtype=np.float32)
        y_test = df_test[label_col].astype(int).to_numpy()
        metrics["test_accuracy"] = float(accuracy_score(y_test, clf.predict(X_test)))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "metrics": metrics,
            "embedding_col": embedding_col,
            "label_col": label_col,
            "split_col": split_col,
        },
        model_out,
    )

    log_event("train_complete", model_path=str(model_out), **metrics)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sentiment classifier from Iceberg embeddings.")
    parser.add_argument(
        "--iceberg-identifier",
        type=str,
        default="imdb.gold",
        help="Iceberg identifier for training, e.g. 'imdb.gold_train'.",
    )
    parser.add_argument(
        "--test-iceberg-identifier",
        type=str,
        default=None,
        help="Optional separate table for test evaluation, e.g. 'imdb.gold_test'.",
    )
    parser.add_argument("--train-split", type=str, default="train", help="Train split name (requires split column).")
    parser.add_argument(
        "--test-split", type=str, default="test", help="Test split name (optional). Use 'none' to disable."
    )
    parser.add_argument("--embedding-col", type=str, default="embedding", help="Column name holding embedding vectors.")
    parser.add_argument("--label-col", type=str, default="label", help="Column name holding label (0/1).")
    parser.add_argument(
        "--split-col", type=str, default="split", help="Column name holding split (train/test/inference)."
    )
    parser.add_argument("--max-records", type=int, default=5000, help="Max rows to sample for training/testing.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--model-out", type=str, default="models/sentiment_logreg.joblib", help="Output model path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    test_split = None if args.test_split.lower() == "none" else args.test_split

    train_and_save(
        iceberg_identifier=args.iceberg_identifier,
        train_split=args.train_split,
        test_split=test_split,
        test_iceberg_identifier=args.test_iceberg_identifier,
        embedding_col=args.embedding_col,
        label_col=args.label_col,
        split_col=args.split_col,
        max_records=None if args.max_records <= 0 else args.max_records,
        random_seed=args.random_seed,
        model_out=Path(args.model_out),
    )


if __name__ == "__main__":
    main()
