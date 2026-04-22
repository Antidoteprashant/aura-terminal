"""Full RAG pipeline: embed query → retrieve → generate → stream."""

import json
import logging
from typing import Generator

logger = logging.getLogger(__name__)


def _sse(data: dict) -> str:
    """Format a dict as a single SSE event string."""
    return f"data: {json.dumps(data)}\n\n"


def query(question: str, conversation_id: str) -> Generator[str, None, None]:
    """Retrieve relevant chunks and stream an LLM answer as SSE events.

    Pipeline:
        1. Embed the question.
        2. Search ChromaDB for top-5 similar chunks.
        3. Format context from retrieved chunks (doc name + text).
        4. Fetch conversation history for conversation_id.
        5. Build prompt via prompts.build_qa_prompt.
        6. Stream response from Ollama, yielding each token as an SSE event.
        7. Save Q&A pair to conversation history.
        8. Yield a final SSE event with done=true and source list.

    Args:
        question: The user's natural-language question.
        conversation_id: Session identifier for conversation memory.

    Yields:
        str: SSE-formatted event strings.
    """
    from app.ingestion import embedder
    from app.ingestion import store as vector_store
    from app.rag import llm, prompts
    from app.models import conversation

    # 1. Embed question
    try:
        q_embedding = embedder.get_embeddings([question])[0]
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        yield _sse({"token": f"[ERROR: Could not embed question — {exc}]"})
        yield _sse({"done": True, "sources": []})
        return

    # 2. Retrieve top-5 chunks
    try:
        hits = vector_store.search(q_embedding, top_k=5)
    except Exception as exc:
        logger.error("ChromaDB search failed: %s", exc)
        yield _sse({"token": f"[ERROR: Could not search documents — {exc}]"})
        yield _sse({"done": True, "sources": []})
        return

    if not hits:
        prompt = prompts.build_no_docs_prompt(question)
        for token in llm.generate_stream(prompt):
            yield _sse({"token": token})
        yield _sse({"done": True, "sources": []})
        return

    # 3. Format context
    context_parts = []
    sources = []
    seen_docs = set()
    for hit in hits:
        filename = hit["metadata"].get("filename", "unknown")
        context_parts.append(f"[{filename}]: {hit['text']}")
        doc_id = hit["metadata"].get("doc_id", "")
        if doc_id not in seen_docs:
            sources.append({"doc_id": doc_id, "filename": filename})
            seen_docs.add(doc_id)
    context = "\n\n".join(context_parts)

    # 4. Conversation history
    history = conversation.get_history(conversation_id)

    # 5. Build prompt
    prompt = prompts.build_qa_prompt(context, history, question)

    # 6. Stream from active LLM backend (Ollama → Groq fallback)
    full_answer = []
    for token in llm.generate_stream(prompt):
        full_answer.append(token)
        yield _sse({"token": token})

    # 7. Save to conversation history
    answer_text = "".join(full_answer)
    conversation.add_message(conversation_id, "user", question)
    conversation.add_message(conversation_id, "assistant", answer_text)

    # 8. Final event with sources
    yield _sse({"done": True, "sources": sources})


def summarize(doc_id: str) -> Generator[str, None, None]:
    """Retrieve all chunks for a document and stream a summary as SSE events.

    Pipeline:
        1. Fetch all chunks for doc_id from ChromaDB.
        2. Build prompt via prompts.build_summarize_prompt.
        3. Stream response from Ollama, yielding each token as an SSE event.
        4. Yield a final SSE done event.

    Args:
        doc_id: The document identifier to summarize.

    Yields:
        str: SSE-formatted event strings.
    """
    from app.ingestion import store as vector_store
    from app.rag import llm, prompts

    # 1. Fetch all chunks for the document
    chunks = vector_store.get_document_chunks(doc_id)

    if not chunks:
        yield _sse({"token": f"[ERROR: Document '{doc_id}' not found or has no content.]"})
        yield _sse({"done": True})
        return

    doc_name = chunks[0]["metadata"].get("filename", doc_id)
    content = "\n\n".join(c["text"] for c in chunks)

    # 2. Build prompt
    prompt = prompts.build_summarize_prompt(doc_name, content)

    # 3. Stream from active LLM backend
    for token in llm.generate_stream(prompt):
        yield _sse({"token": token})

    # 4. Final event
    yield _sse({"done": True})
