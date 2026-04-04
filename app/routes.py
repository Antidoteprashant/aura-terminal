"""API endpoint definitions for the RAG Terminal app."""

import logging
import os
import uuid
from datetime import datetime

from flask import Blueprint, current_app, jsonify, request, Response, stream_with_context

logger = logging.getLogger(__name__)

bp = Blueprint("api", __name__, url_prefix="/api")


# ── Helpers ───────────────────────────────────────────────────

def _allowed_extension(filename: str) -> bool:
    """Return True if the file's extension is in ALLOWED_EXTENSIONS."""
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


def _safe_filename(original: str) -> str:
    """Return a collision-safe temp filename that stays in uploads/.

    Strips directory components and prepends a UUID prefix so concurrent
    uploads of the same filename don't overwrite each other.
    """
    base = os.path.basename(original).replace("\x00", "")  # strip nulls
    return f"{uuid.uuid4().hex}_{base}"


# ── Error handlers ────────────────────────────────────────────

@bp.errorhandler(413)
def request_entity_too_large(e):
    """Return a JSON 413 when the uploaded file exceeds MAX_CONTENT_LENGTH."""
    max_mb = current_app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024) // (1024 * 1024)
    return (
        jsonify({"status": "error", "message": f"File too large. Maximum size is {max_mb} MB."}),
        413,
    )


# ── Endpoints ─────────────────────────────────────────────────

@bp.route("/upload", methods=["POST"])
def upload():
    """Upload a document, parse it, chunk it, embed it, and store in ChromaDB.

    Returns:
        JSON: {"status": "success", "doc_id": str, "filename": str, "chunks": int}
    """
    from app.ingestion import parser, chunker, embedder
    from app.ingestion import store as vector_store

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file part in request."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"status": "error", "message": "No file selected."}), 400

    if not _allowed_extension(file.filename):
        ext = os.path.splitext(file.filename)[1] or "(none)"
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Unsupported file type '{ext}'. Allowed: pdf, txt, md.",
                }
            ),
            415,
        )

    # Use a unique temp name to prevent concurrent-upload collisions
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    safe_name = os.path.basename(file.filename)          # display name
    tmp_name  = _safe_filename(file.filename)            # on-disk name
    filepath  = os.path.join(upload_folder, tmp_name)
    file.save(filepath)

    try:
        # ── Parse ────────────────────────────────────────────
        try:
            parsed = parser.parse_file(filepath)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 422

        text = parsed["text"]

        if not text.strip():
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": (
                            f"'{safe_name}' appears to be empty or contains no "
                            "extractable text."
                        ),
                    }
                ),
                422,
            )

        # ── Chunk ────────────────────────────────────────────
        try:
            chunks = chunker.chunk_text(
                text,
                chunk_size=current_app.config["CHUNK_SIZE"],
                chunk_overlap=current_app.config["CHUNK_OVERLAP"],
            )
        except Exception as exc:
            logger.error("Chunking failed for '%s': %s", safe_name, exc)
            return jsonify({"status": "error", "message": f"Chunking failed: {exc}"}), 500

        if not chunks:
            return (
                jsonify(
                    {"status": "error", "message": "Could not split document into chunks."}
                ),
                422,
            )

        # ── Embed ────────────────────────────────────────────
        try:
            chunk_texts = [c["text"] for c in chunks]
            embeddings  = embedder.get_embeddings(chunk_texts)
        except Exception as exc:
            logger.error("Embedding failed for '%s': %s", safe_name, exc)
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Embedding failed: {exc}. Is the model downloaded?",
                    }
                ),
                500,
            )

        # ── Store ────────────────────────────────────────────
        doc_id = str(uuid.uuid4())
        metadata = {
            "filename":    safe_name,
            "upload_date": datetime.utcnow().strftime("%Y-%m-%d"),
        }
        try:
            vector_store.add_document(doc_id, chunks, embeddings, metadata)
        except Exception as exc:
            logger.error("ChromaDB store failed for '%s': %s", safe_name, exc)
            return (
                jsonify({"status": "error", "message": f"Storage failed: {exc}"}),
                500,
            )

        return jsonify(
            {
                "status":   "success",
                "doc_id":   doc_id,
                "filename": safe_name,
                "chunks":   len(chunks),
            }
        )

    finally:
        # Always clean up the temp file, even on error
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError as exc:
            logger.warning("Could not remove temp file '%s': %s", filepath, exc)


@bp.route("/query", methods=["POST"])
def query():
    """Ask a question over uploaded documents. Returns an SSE stream.

    Request JSON: {"question": str, "conversation_id": str}

    Yields:
        SSE events: data: {"token": str}
        Final event: data: {"done": true, "sources": [...]}
    """
    from app.rag import rag_chain

    body          = request.get_json(silent=True) or {}
    question      = (body.get("question") or "").strip()
    conversation_id = (body.get("conversation_id") or "default").strip()

    if not question:
        return jsonify({"status": "error", "message": "No question provided."}), 400

    @stream_with_context
    def stream():
        yield from rag_chain.query(question, conversation_id)

    return Response(
        stream(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/summarize", methods=["POST"])
def summarize():
    """Summarize a specific document. Returns an SSE stream.

    Request JSON: {"doc_id": str}

    Yields:
        SSE events: data: {"token": str}
        Final event: data: {"done": true}
    """
    from app.rag import rag_chain

    body   = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip()

    if not doc_id:
        return jsonify({"status": "error", "message": "No doc_id provided."}), 400

    @stream_with_context
    def stream():
        yield from rag_chain.summarize(doc_id)

    return Response(
        stream(),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/documents", methods=["GET"])
def list_documents():
    """List all uploaded documents.

    Returns:
        JSON: {"documents": [{"id": str, "filename": str, "upload_date": str, "chunks": int}]}
    """
    from app.ingestion import store as vector_store

    try:
        docs = vector_store.list_documents()
        return jsonify({"documents": docs})
    except Exception as exc:
        logger.error("list_documents error: %s", exc)
        return jsonify({"status": "error", "message": f"Could not list documents: {exc}"}), 500


@bp.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """Delete a document and its embeddings from ChromaDB.

    Returns:
        JSON: {"status": "deleted"}
    """
    from app.ingestion import store as vector_store

    try:
        vector_store.delete_document(doc_id)
        return jsonify({"status": "deleted"})
    except Exception as exc:
        logger.error("delete_document error for '%s': %s", doc_id, exc)
        return jsonify({"status": "error", "message": f"Delete failed: {exc}"}), 500


@bp.route("/health", methods=["GET"])
def health():
    """Check LLM backend availability and system status.

    Returns:
        JSON: {
            "backend": "ollama"|"groq"|"none",
            "ollama": bool,
            "groq": bool,
            "model": str,
            "documents": int,
            "status": "ok"|"error"
        }
    """
    from app.ingestion import store as vector_store
    from app.rag.ollama_client import check_connection as ollama_check
    from app.rag.groq_client import check_connection as groq_check, has_api_key
    from app.rag.llm import resolve_backend

    is_hosted = current_app.config.get("IS_HOSTED", False)

    # In hosted mode skip local Ollama check (it will always be offline)
    ollama_ok = False if is_hosted else ollama_check()
    groq_ok   = groq_check() if has_api_key() else False

    backend, model = resolve_backend()

    # Normalise groq_langchain → groq for the frontend badge
    display_backend = "groq" if backend == "groq_langchain" else backend

    try:
        doc_count = len(vector_store.list_documents())
    except Exception:
        doc_count = 0

    return jsonify(
        {
            "backend":   display_backend,
            "ollama":    ollama_ok,
            "groq":      groq_ok,
            "model":     model,
            "documents": doc_count,
            "status":    "ok" if backend != "none" else "error",
        }
    )


@bp.route("/conversations", methods=["DELETE"])
def clear_conversations():
    """Clear all in-memory conversation history.

    Returns:
        JSON: {"status": "cleared"}
    """
    from app.models.conversation import clear_all

    clear_all()
    return jsonify({"status": "cleared"})
