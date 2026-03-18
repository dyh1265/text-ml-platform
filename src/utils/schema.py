"""Data schemas used across the text-ml-platform project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal


LabelType = Literal[0, 1]


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
    """

    id: str
    text: str
    label: LabelType

    @classmethod
    def from_raw_imdb(cls, raw: Dict[str, Any], idx: int) -> "ImdbBronzeReview":
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

        return cls(id=str(idx), text=text, label=label_raw)

    def to_message(self) -> Dict[str, Any]:
        """Serialise to a dict suitable for sending via Kafka as JSON."""
        return {"id": self.id, "text": self.text, "label": int(self.label)}


@dataclass
class ImdbSilverReview:
    """Schema for a cleaned IMDb review in the silver layer.

    Same structure as bronze but with normalized/cleaned text.
    """

    id: str
    text: str  # cleaned text
    label: LabelType

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for JSONL output."""
        return {"id": self.id, "text": self.text, "label": int(self.label)}


__all__ = ["ImdbBronzeReview", "ImdbSilverReview", "LabelType"]
