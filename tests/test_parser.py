"""Tests for app/ingestion/parser.py"""

import pytest
from app.ingestion.parser import parse_file


def test_parse_txt(tmp_path):
    """parse_file should extract text from a .txt file."""
    pass


def test_parse_pdf(tmp_path):
    """parse_file should extract text from a .pdf file."""
    pass


def test_parse_markdown(tmp_path):
    """parse_file should strip Markdown formatting and return plain text."""
    pass


def test_unsupported_extension(tmp_path):
    """parse_file should raise ValueError for unsupported file types."""
    pass


def test_empty_file(tmp_path):
    """parse_file should handle empty files gracefully."""
    pass
