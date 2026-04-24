"""Full RAG pipeline: embed query → retrieve → generate → stream.

Includes contextual query rewriting so that follow-up questions like
"explain it" or "tell me more" are rewritten into standalone queries
before vector retrieval.
"""

import json
import logging
import re
from typing import Generator

logger = logging.getLogger(__name__)


def _sse(data: dict) -> str:
    """Format a dict as a single SSE event string."""
    return f"data: {json.dumps(data)}\n\n"


# ── Contextual Query Rewriting ────────────────────────────────────────────────

# Short/vague patterns that almost certainly need rewriting
_FOLLOWUP_PATTERNS = re.compile(
    r"^(explain|elaborate|tell me more|what about|why|how|"
    r"can you explain|go on|continue|more details|"
    r"what do you mean|what is that|what are those|"
    r"expand on|clarify|describe|summarize|"
    r"and |but |so |also |then )",
    re.IGNORECASE,
)

# Pronouns that suggest a reference to prior context
_PRONOUN_REFS = re.compile(
    r"\b(it|this|that|these|those|its|them|they|the above|the same)\b",
    re.IGNORECASE,
)


def _needs_rewrite(question: str, history: str) -> bool:
    """Heuristic check: does this question need context from history?

    Returns True if the question is likely a follow-up that references
    prior conversation context (e.g. "explain it", "tell me more").

    Args:
        question: The user's raw question.
        history: Conversation history string (empty if no history).

    Returns:
        bool: True if the question should be rewritten.
    """
    if not history:
        return False  # No history → nothing to reference

    q = question.strip()

    # Very short questions are almost always follow-ups
    if len(q.split()) <= 3:
        return True

    # Starts with a follow-up verb/phrase
    if _FOLLOWUP_PATTERNS.match(q):
        return True

    # Contains pronouns that reference prior context
    if _PRONOUN_REFS.search(q):
        return True

    return False


def _rewrite_query(question: str, history: str) -> str:
    """Use the LLM to rewrite a follow-up question into a standalone query.

    Args:
        question: The user's raw follow-up question.
        history: Conversation history string.

    Returns:
        str: The rewritten standalone question, or the original if rewriting fails.
    """
    from app.rag import llm, prompts

    rewrite_prompt = prompts.build_rewrite_prompt(history, question)

    try:
        rewritten = llm.generate(rewrite_prompt)
    except Exception as exc:
        logger.warning("Query rewrite failed: %s — using original question", exc)
        return question

    # Validate: must be non-empty and not a full answer (guard against LLM misbehaviour)
    rewritten = rewritten.strip().strip('"').strip("'").strip()
    if not rewritten or len(rewritten) > 500:
        logger.warning("Query rewrite produced invalid output — using original")
        return question

    # Strip any trailing period (LLMs sometimes add one)
    if rewritten.endswith("."):
        rewritten = rewritten[:-1].strip()

    logger.info("Query rewritten: '%s' → '%s'", question, rewritten)
    return rewritten


# ── Main Query Pipeline ──────────────────────────────────────────────────────

def query(question: str, conversation_id: str) -> Generator[str, None, None]:
    """Retrieve relevant chunks and stream an LLM answer as SSE events.

    Pipeline:
        0. Check conversation history — if this looks like a follow-up,
           rewrite the question into a standalone query for better retrieval.
        1. Embed the (possibly rewritten) question.
        2. Search ChromaDB for top-5 similar chunks.
        3. Format context from retrieved chunks (doc name + text).
        4. Fetch conversation history for conversation_id.
        5. Build prompt via prompts.build_qa_prompt (using ORIGINAL question).
        6. Stream response from LLM backend, yielding each token as SSE.
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

    # 0. Contextual rewrite for follow-up questions
    history = conversation.get_history(conversation_id)
    search_query = question  # default: use as-is

    if _needs_rewrite(question, history):
        rewritten = _rewrite_query(question, history)
        if rewritten and rewritten != question:
            search_query = rewritten
            # Let the user know the query was understood contextually
            yield _sse({"context_note": f"Understood as: {search_query}"})

    # 1. Embed the search query (rewritten or original)
    try:
        q_embedding = embedder.get_embeddings([search_query])[0]
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

    # 4. Build prompt — always use the ORIGINAL question so the answer reads
    #    naturally, but use the REWRITTEN query for retrieval above.
    prompt = prompts.build_qa_prompt(context, history, question)

    # 5. Stream from active LLM backend
    full_answer = []
    for token in llm.generate_stream(prompt):
        full_answer.append(token)
        yield _sse({"token": token})

    # 6. Save to conversation history
    answer_text = "".join(full_answer)
    conversation.add_message(conversation_id, "user", question)
    conversation.add_message(conversation_id, "assistant", answer_text)

    # 7. Save source citations for this turn
    conversation.add_citation(conversation_id, question, sources)

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
