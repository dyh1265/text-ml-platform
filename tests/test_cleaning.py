"""Tests for cleaning module."""

import pytest

from src.transformation.cleaning import clean_text, normalize_whitespace, strip_html


def test_strip_html():
    assert strip_html("<p>Hello</p>") == " Hello "
    assert strip_html("No tags here") == "No tags here"
    # Two tags become two spaces; content preserved
    assert strip_html("<br/><div>Hi</div>").strip() == "Hi"


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("a\tb\nc") == "a b c"


def test_clean_text_default():
    assert clean_text("  Hello   World  ") == "hello world"
    assert clean_text("<p>GREAT film!</p>") == "great film!"


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_no_lowercase():
    result = clean_text("Hello World", lowercase=False)
    assert result == "Hello World"


def test_clean_text_no_strip_html():
    result = clean_text("<b>bold</b>", strip_html_tags=False)
    assert "<b>" in result and "</b>" in result
