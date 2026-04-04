"""In-memory conversation history keyed by conversation_id."""

# {conversation_id: [{"role": "user"|"assistant", "content": str}]}
_history: dict[str, list[dict]] = {}

MAX_TURNS = 5


def add_message(conv_id: str, role: str, content: str) -> None:
    """Append a message to the conversation history.

    Keeps at most MAX_TURNS * 2 messages (5 user + 5 assistant) to stay
    within the model's context window. Oldest messages are dropped first.

    Args:
        conv_id: Conversation session identifier.
        role: "user" or "assistant".
        content: The message text.
    """
    if conv_id not in _history:
        _history[conv_id] = []

    _history[conv_id].append({"role": role, "content": content})

    # Keep only the last MAX_TURNS pairs (2 messages per turn)
    max_messages = MAX_TURNS * 2
    if len(_history[conv_id]) > max_messages:
        _history[conv_id] = _history[conv_id][-max_messages:]


def get_history(conv_id: str, max_turns: int = MAX_TURNS) -> str:
    """Return the formatted conversation history as a string.

    Format: "User: ...\nAssistant: ..."

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


def clear(conv_id: str) -> None:
    """Clear the history for a specific conversation.

    Args:
        conv_id: Conversation session identifier.
    """
    _history.pop(conv_id, None)


def clear_all() -> None:
    """Clear all conversation history across all sessions."""
    _history.clear()
