"""Text cleaning utilities for the silver layer."""

from __future__ import annotations

import re


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace and strip leading/trailing."""
    return " ".join(text.split())


def clean_text(
    text: str,
    lowercase: bool = True,
    strip_html_tags: bool = True,
    normalize_whitespace_flag: bool = True,
) -> str:
    """Clean and normalize raw review text.

    Parameters
    ----------
    text :
        Raw text to clean.
    lowercase :
        Convert to lowercase.
    strip_html_tags :
        Remove HTML tags.
    normalize_whitespace_flag :
        Collapse and trim whitespace.

    Returns
    -------
    Cleaned text.
    """
    if not text:
        return ""

    s = str(text)
    if strip_html_tags:
        s = strip_html(s)
    if normalize_whitespace_flag:
        s = normalize_whitespace(s)
    if lowercase:
        s = s.lower()

    return s.strip()


__all__ = ["clean_text", "normalize_whitespace", "strip_html"]
