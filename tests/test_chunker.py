"""Tests for app/ingestion/chunker.py"""

import pytest
from app.ingestion.chunker import chunk_text


def test_short_text_single_chunk():
    """Text shorter than chunk_size should return a single chunk."""
    pass


def test_long_text_multiple_chunks():
    """Text longer than chunk_size should be split into multiple chunks."""
    pass


def test_chunk_overlap():
    """Chunks should overlap by the specified amount."""
    pass


def test_chunk_index_sequential():
    """chunk_index values should be sequential starting from 0."""
    pass
