"""Tests for schema module."""

from src.utils.schema import ImdbBronzeReview


def test_imdb_bronze_review_from_raw_imdb_happy_path():
    raw = {"text": "Great movie!", "label": 1}
    review = ImdbBronzeReview.from_raw_imdb(raw, idx=42)

    assert review.id == "42"
    assert review.text == "Great movie!"
    assert review.label == 1


def test_imdb_bronze_review_from_raw_imdb_rejects_bad_label():
    raw = {"text": "meh", "label": 2}

    try:
        ImdbBronzeReview.from_raw_imdb(raw, idx=0)
    except ValueError as exc:
        assert "Unexpected IMDb label value" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError for invalid label")


def test_imdb_bronze_review_to_message_roundtrip_keys():
    review = ImdbBronzeReview(id="abc", text="ok", label=0)
    msg = review.to_message()

    assert msg["id"] == "abc"
    assert msg["text"] == "ok"
    assert msg["label"] == 0
