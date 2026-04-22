"""Prompt templates for the RAG Q&A and summarization pipelines."""


def build_rewrite_prompt(history: str, question: str) -> str:
    """Build a prompt that rewrites a follow-up question into a standalone query.

    When the user asks something like "explain it" or "tell me more", the raw
    question is too vague for vector search. This prompt asks the LLM to
    produce a single, self-contained question using conversation context.

    Args:
        history: Conversation history formatted as "User: ...\\nAssistant: ...".
        question: The user's current (possibly vague) follow-up question.

    Returns:
        str: The fully-formatted rewrite prompt.
    """
    return (
        "You are a query rewriter. Given the conversation history and a follow-up "
        "question, rewrite the follow-up into a standalone question that captures "
        "the full intent.\n"
        "\n"
        "RULES:\n"
        "- Output ONLY the rewritten question, nothing else\n"
        "- Do NOT answer the question\n"
        "- If the question is already standalone, return it unchanged\n"
        "- Preserve the original intent and specificity\n"
        "- Include relevant context from the conversation history\n"
        "\n"
        f"CONVERSATION HISTORY:\n{history}\n"
        "\n"
        f"FOLLOW-UP QUESTION: {question}\n"
        "\n"
        "STANDALONE QUESTION:"
    )



def build_qa_prompt(context: str, history: str, question: str) -> str:
    """Build the QA prompt with retrieved context, conversation history, and question.

    Uses a hybrid approach: the LLM first answers from document context,
    then supplements with its own knowledge when documents lack detail.

    Args:
        context: Retrieved document chunks, formatted as "doc_name: chunk_text".
        history: Conversation history formatted as "User: ...\\nAssistant: ...".
        question: The user's current question.

    Returns:
        str: The fully-formatted prompt string ready to send to the LLM.
    """
    return (
        "You are a helpful study assistant with access to the user's uploaded documents "
        "and your own general knowledge.\n"
        "\n"
        "RULES:\n"
        "- FIRST, answer using the provided document context below\n"
        "- Cite the document name when using information from the context "
        "(e.g. 'According to [filename]...')\n"
        "- If the documents mention a topic but don't explain it in detail, "
        "use your own knowledge to elaborate and explain it further\n"
        "- When using your own knowledge beyond the documents, briefly note it "
        "(e.g. 'Based on general knowledge...' or 'To elaborate further...')\n"
        "- If the question is a follow-up, use the conversation history for context\n"
        "- Be thorough but well-structured — use headings, bullet points, or "
        "numbered lists when appropriate\n"
        "- If the documents have NO relevant information at all and you cannot "
        "answer from general knowledge either, say so clearly\n"
        "\n"
        f"DOCUMENT CONTEXT:\n{context}\n"
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
