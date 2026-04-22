"""In-memory conversation history and source citation tracking.

Stores per-session:
  - Message history (user/assistant turns) for follow-up context
  - Source citations (which documents were used to answer each question)
"""

from datetime import datetime

# {conversation_id: [{"role": "user"|"assistant", "content": str, "timestamp": str}]}
_history: dict[str, list[dict]] = {}

# {conversation_id: [{"question": str, "sources": [...], "timestamp": str}]}
_citations: dict[str, list[dict]] = {}

MAX_TURNS = 10  # 10 turns of context for better follow-up support


def add_message(conv_id: str, role: str, content: str) -> None:
    """Append a message to the conversation history.

    Keeps at most MAX_TURNS * 2 messages (user + assistant pairs) to stay
    within the model's context window. Oldest messages are dropped first.

    Args:
        conv_id: Conversation session identifier.
        role: "user" or "assistant".
        content: The message text.
    """
    if conv_id not in _history:
        _history[conv_id] = []

    _history[conv_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
    })

    # Keep only the last MAX_TURNS pairs (2 messages per turn)
    max_messages = MAX_TURNS * 2
    if len(_history[conv_id]) > max_messages:
        _history[conv_id] = _history[conv_id][-max_messages:]


def get_history(conv_id: str, max_turns: int = MAX_TURNS) -> str:
    """Return the formatted conversation history as a string.

    Format: "User: ...\\nAssistant: ..."

    Args:
        conv_id: Conversation session identifier.
        max_turns: Maximum number of turns to include.

    Returns:
        str: Formatted history, or empty string if no history exists.
    """
    messages = _history.get(conv_id, [])
    if not messages:
        return ""

    # Take the last max_turns pairs
    messages = messages[-(max_turns * 2):]

    lines = []
    for msg in messages:
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{label}: {msg['content']}")

    return "\n".join(lines)


def get_raw_history(conv_id: str) -> list[dict]:
    """Return the raw message list for a conversation.

    Args:
        conv_id: Conversation session identifier.

    Returns:
        list[dict]: List of message dicts with role, content, timestamp.
    """
    return _history.get(conv_id, [])


# ── Source Citation Tracking ─────────────────────────────────────────────────

def add_citation(conv_id: str, question: str, sources: list[dict]) -> None:
    """Record which documents were cited when answering a question.

    Args:
        conv_id: Conversation session identifier.
        question: The user's question that was answered.
        sources: List of source dicts, each with "doc_id" and "filename".
    """
    if conv_id not in _citations:
        _citations[conv_id] = []

    _citations[conv_id].append({
        "question": question,
        "sources": sources,
        "timestamp": datetime.utcnow().strftime("%H:%M:%S"),
    })


def get_citations(conv_id: str) -> list[dict]:
    """Return all source citations for a conversation session.

    Args:
        conv_id: Conversation session identifier.

    Returns:
        list[dict]: List of citation records, each with question, sources, timestamp.
    """
    return _citations.get(conv_id, [])


def get_all_cited_documents(conv_id: str) -> list[dict]:
    """Return a deduplicated list of all documents cited in a conversation.

    Args:
        conv_id: Conversation session identifier.

    Returns:
        list[dict]: Unique documents cited, each with "doc_id", "filename", "times_cited".
    """
    citations = _citations.get(conv_id, [])
    doc_counts: dict[str, dict] = {}

    for entry in citations:
        for src in entry["sources"]:
            did = src.get("doc_id", "")
            if did in doc_counts:
                doc_counts[did]["times_cited"] += 1
            else:
                doc_counts[did] = {
                    "doc_id": did,
                    "filename": src.get("filename", "unknown"),
                    "times_cited": 1,
                }

    return list(doc_counts.values())


def clear(conv_id: str) -> None:
    """Clear the history and citations for a specific conversation.

    Args:
        conv_id: Conversation session identifier.
    """
    _history.pop(conv_id, None)
    _citations.pop(conv_id, None)


def clear_all() -> None:
    """Clear all conversation history and citations across all sessions."""
    _history.clear()
    _citations.clear()
