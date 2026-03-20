"""Tests for producer."""

from unittest.mock import patch

from src.ingestion.producer import load_imdb_reviews


def test_load_imdb_reviews():
    """load_imdb_reviews yields ImdbBronzeReview from mocked dataset."""
    mock_rows = [
        {"text": "great movie", "label": 1},
        {"text": "awful", "label": 0},
    ]

    class MockDs:
        def __init__(self, data):
            self._data = data

        def shuffle(self, seed=None):
            return self

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

    with patch("src.ingestion.producer.load_dataset", return_value=MockDs(mock_rows)):
        reviews = list(load_imdb_reviews(split="train", limit=2, shuffle=False))
        assert len(reviews) == 2
        assert reviews[0].text == "great movie" and reviews[0].label == 1
        assert reviews[1].text == "awful" and reviews[1].label == 0
