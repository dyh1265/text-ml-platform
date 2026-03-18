"""Tests for silver job."""

from src.transformation.cleaning import clean_text
from src.transformation.silver_job import deduplicate_by_text_label
from src.utils.schema import ImdbSilverReview


def test_deduplicate_by_text_label_removes_duplicates():
    """Deduplication keeps first occurrence of each (text, label) pair."""
    recs = [
        ImdbSilverReview(id="1", text="same text", label=1),
        ImdbSilverReview(id="2", text="same text", label=1),
        ImdbSilverReview(id="3", text="other text", label=0),
        ImdbSilverReview(id="4", text="same text", label=0),
    ]
    result = deduplicate_by_text_label(recs)
    assert len(result) == 3
    assert result[0].id == "1"
    assert result[1].id == "3"
    assert result[2].id == "4"


def test_deduplicate_by_text_label_cross_batch():
    """Shared seen set enables cross-batch deduplication."""
    seen = set()
    batch1 = [
        ImdbSilverReview(id="1", text="dup", label=1),
        ImdbSilverReview(id="2", text="unique", label=0),
    ]
    batch2 = [
        ImdbSilverReview(id="3", text="dup", label=1),
    ]
    r1 = deduplicate_by_text_label(batch1, seen=seen)
    r2 = deduplicate_by_text_label(batch2, seen=seen)
    assert len(r1) == 2
    assert len(r2) == 0


def test_imdb_silver_review_to_dict():
    silver = ImdbSilverReview(id="1", text="cleaned text", label=1)
    d = silver.to_dict()
    assert d["id"] == "1"
    assert d["text"] == "cleaned text"
    assert d["label"] == 1


def test_cleaning_integration():
    raw = "  <p>GREAT   MOVIE!</p>  "
    cleaned = clean_text(raw)
    assert cleaned == "great movie!"
    assert "<" not in cleaned
    assert "  " not in cleaned
