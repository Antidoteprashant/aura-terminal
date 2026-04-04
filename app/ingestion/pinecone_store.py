"""Pinecone vector store — hosted alternative to ChromaDB.

Implements the same public interface as vector_store.py so routes and the
RAG chain can call it without knowing which backend is active.

Pinecone stores vectors with metadata. Since Pinecone has no native
"list all documents" scan, we reconstruct the document list by querying
with a zero-vector and grouping results on the ``doc_id`` metadata field.
"""

import logging
from collections import defaultdict
from typing import Generator

logger = logging.getLogger(__name__)

# Module-level Pinecone index — initialized once by init_store()
_index = None
_embed_dim: int = 384          # all-MiniLM-L6-v2 output dimension


def _cfg(key: str, default: str = "") -> str:
    try:
        from flask import current_app
        return current_app.config.get(key, default)
    except RuntimeError:
        import os
        return os.getenv(key, default)


def init_store() -> None:
    """Connect to the Pinecone index specified in config.

    The index must already exist in your Pinecone project. If it does not
    exist, this function creates a serverless index (us-east-1, AWS) with
    cosine similarity and dimension 384 (matching all-MiniLM-L6-v2).
    """
    global _index

    if _index is not None:
        return

    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        raise RuntimeError(
            "pinecone-client is not installed. "
            "Run: pip install pinecone-client"
        )

    api_key    = _cfg("PINECONE_API_KEY")
    index_name = _cfg("PINECONE_INDEX", "rag-terminal")

    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. "
            "Add it to your environment variables."
        )

    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist yet
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info("Creating Pinecone index '%s' (dim=%d, cosine).", index_name, _embed_dim)
        pc.create_index(
            name=index_name,
            dimension=_embed_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    _index = pc.Index(index_name)
    logger.info("Pinecone index '%s' ready.", index_name)


def _get_index():
    if _index is None:
        init_store()
    return _index


def add_document(
    doc_id: str, chunks: list[dict], embeddings: list, metadata: dict
) -> None:
    """Upsert a document's chunks and embeddings into Pinecone.

    Args:
        doc_id: Unique document identifier.
        chunks: List of chunk dicts from chunker.chunk_text.
        embeddings: List of embedding vectors (384-dim each).
        metadata: Document-level metadata (filename, upload_date, etc.).
    """
    index = _get_index()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": f"{doc_id}_{chunk['chunk_index']}",
            "values": embedding,
            "metadata": {
                "doc_id":      doc_id,
                "filename":    metadata.get("filename", ""),
                "upload_date": metadata.get("upload_date", ""),
                "chunk_index": chunk["chunk_index"],
                "start_char":  chunk["start_char"],
                "text":        chunk["text"],   # stored for retrieval
            },
        })

    # Pinecone recommends batches ≤ 100
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])

    logger.info("Upserted %d chunks for doc_id='%s' to Pinecone.", len(chunks), doc_id)


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Return the top-k most similar chunks from Pinecone.

    Args:
        query_embedding: 384-dimensional query vector.
        top_k: Number of results to return.

    Returns:
        list[dict]: Matches with "text", "metadata", and "distance" keys.
    """
    index = _get_index()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    hits = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        hits.append({
            "text":     meta.get("text", ""),
            "metadata": {k: v for k, v in meta.items() if k != "text"},
            "distance": 1.0 - match.get("score", 0.0),  # cosine: distance = 1 - similarity
        })

    return hits


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to a document from Pinecone.

    Args:
        doc_id: The document identifier to delete.
    """
    index = _get_index()

    # Pinecone's delete with metadata filter (requires serverless or paid tier)
    try:
        index.delete(filter={"doc_id": doc_id})
        logger.info("Deleted all chunks for doc_id='%s' from Pinecone.", doc_id)
    except Exception as exc:
        # Older Pinecone plans may not support metadata filtering on delete;
        # fall back to ID-based delete by listing all IDs with that prefix.
        logger.warning(
            "Metadata-filter delete failed (%s). Falling back to ID-based delete.", exc
        )
        _delete_by_prefix(index, f"{doc_id}_")


def _delete_by_prefix(index, prefix: str) -> None:
    """Delete all vectors whose ID starts with prefix (pagination-safe)."""
    ids_to_delete = []
    for page in index.list(prefix=prefix):
        ids_to_delete.extend(page)
    if ids_to_delete:
        index.delete(ids=ids_to_delete)


def get_document_chunks(doc_id: str) -> list[dict]:
    """Return all chunks for a specific document, ordered by chunk_index.

    Args:
        doc_id: The document identifier.

    Returns:
        list[dict]: Chunk dicts with "text" and "metadata" keys.
    """
    index = _get_index()

    # Collect all vector IDs for this doc via prefix listing
    all_ids = []
    try:
        for page in index.list(prefix=f"{doc_id}_"):
            all_ids.extend(page)
    except Exception:
        # Older SDK: can't list, so query with zero-vector + filter
        import os
        zero = [0.0] * _embed_dim
        results = index.query(
            vector=zero,
            top_k=1000,
            filter={"doc_id": doc_id},
            include_metadata=True,
        )
        chunks = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            chunks.append({
                "text":     meta.get("text", ""),
                "metadata": {k: v for k, v in meta.items() if k != "text"},
            })
        chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
        return chunks

    if not all_ids:
        return []

    # Fetch vectors by ID in batches of 1000 (Pinecone limit)
    chunks = []
    batch_size = 1000
    for i in range(0, len(all_ids), batch_size):
        batch = index.fetch(ids=all_ids[i : i + batch_size])
        for vec_id, vec_data in batch.get("vectors", {}).items():
            meta = vec_data.get("metadata", {})
            chunks.append({
                "text":     meta.get("text", ""),
                "metadata": {k: v for k, v in meta.items() if k != "text"},
            })

    chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    return chunks


def list_documents() -> list[dict]:
    """List all unique documents in Pinecone.

    Uses a zero-vector query (top_k=10000) to discover all stored chunks,
    then groups by doc_id to reconstruct the document list.

    Returns:
        list[dict]: One entry per document: {id, filename, upload_date, chunks}.
    """
    index = _get_index()

    try:
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)
        if total == 0:
            return []
    except Exception:
        total = 1  # unknown — proceed and let query return empty

    zero = [0.0] * _embed_dim
    top_k = min(total, 10000)

    try:
        results = index.query(
            vector=zero,
            top_k=top_k,
            include_metadata=True,
        )
    except Exception as exc:
        logger.error("Pinecone list_documents query failed: %s", exc)
        return []

    docs: dict[str, dict] = {}
    chunk_counts: dict[str, int] = defaultdict(int)

    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        if not doc_id:
            continue
        if doc_id not in docs:
            docs[doc_id] = {
                "id":          doc_id,
                "filename":    meta.get("filename", ""),
                "upload_date": meta.get("upload_date", ""),
            }
        chunk_counts[doc_id] += 1

    return [
        {**doc_info, "chunks": chunk_counts[doc_id]}
        for doc_id, doc_info in docs.items()
    ]
