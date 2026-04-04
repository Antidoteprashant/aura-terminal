"""Prompt templates for the RAG Q&A and summarization pipelines."""


def build_qa_prompt(context: str, history: str, question: str) -> str:
    """Build the QA prompt with retrieved context, conversation history, and question.

    Args:
        context: Retrieved document chunks, formatted as "doc_name: chunk_text".
        history: Conversation history formatted as "User: ...\nAssistant: ...".
        question: The user's current question.

    Returns:
        str: The fully-formatted prompt string ready to send to the LLM.
    """
    return (
        "You are a helpful study assistant. Answer the question using ONLY the provided context.\n"
        "If the context doesn't contain enough information, say "
        '"I don\'t have enough information in the uploaded documents to answer this."\n'
        "\n"
        "RULES:\n"
        "- Only use information from the context below\n"
        "- Cite your sources by mentioning the document name\n"
        "- Be concise but thorough\n"
        "- If the question is a follow-up, use the conversation history for context\n"
        "\n"
        f"CONTEXT:\n{context}\n"
        "\n"
        f"CONVERSATION HISTORY:\n{history}\n"
        "\n"
        f"QUESTION: {question}\n"
        "\n"
        "ANSWER:"
    )


def build_summarize_prompt(doc_name: str, content: str) -> str:
    """Build the summarization prompt for a document.

    Args:
        doc_name: The filename of the document.
        content: The full document text (or concatenated chunks).

    Returns:
        str: The fully-formatted summarization prompt.
    """
    return (
        "You are a study assistant. Provide a clear, concise summary of the following document content.\n"
        "Focus on key concepts, definitions, and important points that a student would need to know.\n"
        "\n"
        f"DOCUMENT: {doc_name}\n"
        "CONTENT:\n"
        f"{content}\n"
        "\n"
        "SUMMARY:"
    )
