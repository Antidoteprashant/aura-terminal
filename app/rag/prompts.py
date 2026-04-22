"""Prompt templates for the RAG Q&A and summarization pipelines."""

# ── Token Budget Guards ───────────────────────────────────────
# Max character budgets (rough approximation: 4 chars ~ 1 token)
MAX_CONTEXT_CHARS = 12000  # ~3000 tokens
MAX_HISTORY_CHARS = 2000   # ~500 tokens


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to fit within a character budget, adding an ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for context length]"


# ── Prompt Builders ───────────────────────────────────────────

def build_qa_prompt(context: str, history: str, question: str) -> str:
    """Build a robust QA prompt with character budgeting.

    Args:
        context: Retrieved document chunks.
        history: Conversation history.
        question: The user's current question.
    """
    safe_context = _truncate(context, MAX_CONTEXT_CHARS)
    safe_history = _truncate(history, MAX_HISTORY_CHARS)

    return (
        "SYSTEM: You are AURA, an intelligent RAG Terminal assistant. "
        "Answer the user's question using ONLY the provided CONTEXT. "
        "If the answer isn't in the context, say: 'I don't have enough information in the uploaded documents to answer this.'\n\n"
        
        "RULES:\n"
        "1. Stay concise and technical.\n"
        "2. Use bullet points for lists.\n"
        "3. ALWAYS cite the filename (e.g., [notes.pdf]) when using information from it.\n"
        "4. If the question is a follow-up, use the CONVERSATION HISTORY for context.\n\n"
        
        f"CONTEXT:\n{safe_context}\n\n"
        f"CONVERSATION HISTORY:\n{safe_history}\n\n"
        f"USER: {question}\n\n"
        "ASSISTANT:"
    )


def build_summarize_prompt(doc_name: str, content: str) -> str:
    """Build a structured summarization prompt.

    Args:
        doc_name: Filename.
        content: Document content.
    """
    safe_content = _truncate(content, MAX_CONTEXT_CHARS)

    return (
        "SYSTEM: You are a study assistant. Provide a structured summary of the document below.\n\n"
        
        f"DOCUMENT: {doc_name}\n"
        "CONTENT:\n"
        f"{safe_content}\n\n"
        
        "INSTRUCTIONS:\n"
        "- Start with a 1-sentence overview.\n"
        "- List 5-7 key concepts or definitions as bullet points.\n"
        "- End with a 'Key Takeaway' section.\n\n"
        "SUMMARY:"
    )


def build_no_docs_prompt(question: str) -> str:
    """Prompt for when the user asks a question but no documents are uploaded.
    
    This allows the LLM to explain the system capability rather than just failing.
    """
    return (
        "SYSTEM: You are AURA. The user has not uploaded any documents yet.\n"
        "Briefly explain that you are a RAG (Retrieval-Augmented Generation) terminal "
        "and that they should use the /upload command or drag-and-drop files to get started.\n\n"
        f"USER: {question}\n"
        "ASSISTANT:"
    )
