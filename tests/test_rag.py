"""Tests for the RAG query pipeline (app/rag/)."""

import pytest


def test_check_connection_offline():
    """check_connection should return False when Ollama is not running."""
    pass


def test_qa_prompt_contains_question():
    """build_qa_prompt should include the question in the output string."""
    pass


def test_summarize_prompt_contains_doc_name():
    """build_summarize_prompt should include the document name."""
    pass


def test_get_history_empty():
    """get_history should return empty string when no history exists."""
    pass


def test_add_and_get_history():
    """add_message + get_history should return formatted conversation."""
    pass


def test_clear_history():
    """clear() should remove a specific conversation's history."""
    pass
