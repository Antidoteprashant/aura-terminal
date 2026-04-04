"""Vector store proxy — routes calls to ChromaDB (local) or Pinecone (hosted).

Import this module wherever you need vector-store operations instead of
importing vector_store or pinecone_store directly.  Environment switching
is handled here transparently.

Usage::

    from app.ingestion import store as vector_store
    vector_store.init_store()
    vector_store.add_document(...)
"""


def _backend():
    """Return the active vector-store module based on IS_HOSTED config."""
    try:
        from flask import current_app
        hosted = current_app.config.get("IS_HOSTED", False)
    except RuntimeError:
        import os
        hosted = os.getenv("IS_HOSTED", "").lower() in ("1", "true", "yes")

    if hosted:
        from app.ingestion import pinecone_store
        return pinecone_store
    else:
        from app.ingestion import vector_store
        return vector_store


# ── Public interface (mirrors both backends) ─────────────────────────────────

def init_store() -> None:
    _backend().init_store()


def add_document(doc_id: str, chunks: list, embeddings: list, metadata: dict) -> None:
    _backend().add_document(doc_id, chunks, embeddings, metadata)


def search(query_embedding: list, top_k: int = 5) -> list:
    return _backend().search(query_embedding, top_k)


def delete_document(doc_id: str) -> None:
    _backend().delete_document(doc_id)


def get_document_chunks(doc_id: str) -> list:
    return _backend().get_document_chunks(doc_id)


def list_documents() -> list:
    return _backend().list_documents()
