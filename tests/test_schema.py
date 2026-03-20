"""Tests for schema module."""

import pytest

from src.utils.schema import ImdbBronzeReview, ImdbSilverReview


def test_imdb_bronze_review_from_raw_imdb_happy_path():
    raw = {"text": "Great movie!", "label": 1}
    review = ImdbBronzeReview.from_raw_imdb(raw, idx=42)

    assert review.id == "42"
    assert review.text == "Great movie!"
    assert review.label == 1
    assert review.split == "unknown"
    assert review.request_id is None


def test_imdb_bronze_review_from_raw_imdb_with_split():
    raw = {"text": "Okay film", "label": 0}
    review = ImdbBronzeReview.from_raw_imdb(raw, idx=1, split="train")
    assert review.split == "train"


def test_imdb_bronze_review_from_raw_imdb_rejects_bad_label():
    raw = {"text": "meh", "label": 2}
    with pytest.raises(ValueError, match="Unexpected IMDb label value"):
        ImdbBronzeReview.from_raw_imdb(raw, idx=0)


def test_imdb_bronze_review_to_message_roundtrip_keys():
    review = ImdbBronzeReview(id="abc", text="ok", label=0)
    msg = review.to_message()

    assert msg["id"] == "abc"
    assert msg["text"] == "ok"
    assert msg["label"] == 0
    assert "request_id" not in msg


def test_imdb_bronze_review_to_message_includes_request_id_when_set():
    review = ImdbBronzeReview(id="x", text="hi", label=1, request_id="req-123")
    msg = review.to_message()
    assert msg["request_id"] == "req-123"


def test_imdb_silver_review_to_dict():
    review = ImdbSilverReview(id="s1", text="cleaned text", label=1, split="test")
    d = review.to_dict()
    assert d["id"] == "s1"
    assert d["text"] == "cleaned text"
    assert d["label"] == 1
    assert d["split"] == "test"
    assert "request_id" not in d


def test_imdb_silver_review_to_dict_with_request_id():
    review = ImdbSilverReview(id="s2", text="x", label=0, request_id="r-456")
    d = review.to_dict()
    assert d["request_id"] == "r-456"
