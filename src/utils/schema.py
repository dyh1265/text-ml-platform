"""Data schemas used across the text-ml-platform project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pyarrow as pa
    from pyiceberg.schema import Schema


LabelType = Literal[0, 1]
SplitType = Literal["train", "test", "inference", "unknown"]


@dataclass
class ImdbBronzeReview:
    """Schema for a single IMDb review in the bronze layer.

    Attributes
    ----------
    id:
        A unique identifier for the review (stringified index or UUID).
    text:
        The raw review text.
    label:
        Binary sentiment label: 0 = negative, 1 = positive.
    split:
        Which split this record belongs to (train/test/inference/unknown).
    request_id:
        Optional identifier used to correlate records end-to-end for a single
        UI request (useful for lineage + async inference).
    """

    id: str
    text: str
    label: LabelType
    split: SplitType = "unknown"
    request_id: str | None = None

    @classmethod
    def from_raw_imdb(
        cls,
        raw: dict[str, Any],
        idx: int,
        *,
        split: SplitType = "unknown",
    ) -> ImdbBronzeReview:
        """Create an instance from a raw IMDb dataset row.

        Parameters
        ----------
        raw:
            A single record from the IMDb dataset (expects keys ``text`` and ``label``).
        idx:
            Row index, used as a stable default identifier.
        """

        text = str(raw.get("text", ""))
        label_raw = raw.get("label")

        if label_raw not in (0, 1):
            raise ValueError(f"Unexpected IMDb label value: {label_raw!r}")

        return cls(id=str(idx), text=text, label=label_raw, split=split)

    def to_message(self) -> dict[str, Any]:
        """Serialise to a dict suitable for sending via Kafka as JSON."""
        msg: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "label": int(self.label),
            "split": self.split,
        }
        if self.request_id is not None:
            msg["request_id"] = self.request_id
        return msg


@dataclass
class ImdbSilverReview:
    """Schema for a cleaned IMDb review in the silver layer.

    Same structure as bronze but with normalized/cleaned text.
    """

    id: str
    text: str  # cleaned text
    label: LabelType
    split: SplitType = "unknown"
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSONL output."""
        msg: dict[str, Any] = {"id": self.id, "text": self.text, "label": int(self.label), "split": self.split}
        if self.request_id is not None:
            msg["request_id"] = self.request_id
        return msg


def gold_embedding_iceberg_schema() -> Schema:
    """Return the Iceberg schema for gold embedding tables.

    Shared by ``embedding_job`` and ``inference_worker`` so the schema
    definition lives in one place.
    """
    from pyiceberg.schema import Schema as _Schema
    from pyiceberg.types import FloatType, ListType, LongType, NestedField, StringType

    return _Schema(
        NestedField(1, "id", StringType(), required=True),
        NestedField(2, "text", StringType(), required=True),
        NestedField(3, "label", LongType(), required=True),
        NestedField(
            4,
            "embedding",
            ListType(element_id=5, element_type=FloatType(), element_required=False),
            required=True,
        ),
        NestedField(6, "split", StringType(), required=True),
        NestedField(7, "request_id", StringType(), required=False),
    )


def gold_embedding_arrow_schema() -> pa.schema:
    """Return the PyArrow schema matching :func:`gold_embedding_iceberg_schema`."""
    import pyarrow as _pa

    return _pa.schema(
        [
            _pa.field("id", _pa.string(), nullable=False),
            _pa.field("text", _pa.string(), nullable=False),
            _pa.field("label", _pa.int64(), nullable=False),
            _pa.field("embedding", _pa.list_(_pa.float32()), nullable=False),
            _pa.field("split", _pa.string(), nullable=False),
            _pa.field("request_id", _pa.string(), nullable=True),
        ]
    )


__all__ = [
    "ImdbBronzeReview",
    "ImdbSilverReview",
    "LabelType",
    "SplitType",
    "gold_embedding_arrow_schema",
    "gold_embedding_iceberg_schema",
]
