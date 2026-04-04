"""ChromaDB wrapper for persistent vector storage."""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Module-level ChromaDB client and collection — initialized once
_client = None
_collection = None


def init_store():
    """Create or connect to the persistent ChromaDB store.

    Initializes the module-level client and collection singletons.
    Reads CHROMA_PERSIST_DIR and CHROMA_COLLECTION_NAME from Flask config.
    """
    global _client, _collection

    if _collection is not None:
        return

    import chromadb
    from flask import current_app

    persist_dir = current_app.config["CHROMA_PERSIST_DIR"]
    collection_name = current_app.config["CHROMA_COLLECTION_NAME"]

    logger.info("Initialising ChromaDB at '%s'", persist_dir)
    _client = chromadb.PersistentClient(path=persist_dir)
    _collection = _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection '%s' ready.", collection_name)


def _get_collection():
    """Return the active collection, initialising if needed."""
    if _collection is None:
        init_store()
    return _collection


def add_document(
    doc_id: str, chunks: list[dict], embeddings: list, metadata: dict
) -> None:
    """Add a document's chunks and embeddings to ChromaDB.

    Each chunk is stored with a unique ID of the form ``{doc_id}_{chunk_index}``
    and carries per-chunk metadata so it can be filtered or deleted later.

    Args:
        doc_id: Unique identifier for the document.
        chunks: List of chunk dicts from chunker.chunk_text.
        embeddings: List of embedding vectors from embedder.get_embeddings.
        metadata: Document-level metadata (filename, upload_date, etc.).
    """
    collection = _get_collection()

    ids = [f"{doc_id}_{chunk['chunk_index']}" for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": metadata.get("filename", ""),
            "upload_date": metadata.get("upload_date", ""),
            "chunk_index": chunk["chunk_index"],
            "start_char": chunk["start_char"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    logger.info("Stored %d chunks for doc_id='%s'.", len(chunks), doc_id)


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search for the most similar chunks to a query embedding.

    Args:
        query_embedding: The query vector (384-dimensional).
        top_k: Number of results to return.

    Returns:
        list[dict]: Matching chunks with text, metadata, and distance.
    """
    collection = _get_collection()

    # Clamp top_k to the number of stored items
    count = collection.count()
    if count == 0:
        return []
    top_k = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})

    return hits


def delete_document(doc_id: str) -> None:
    """Remove all chunks belonging to a document from ChromaDB.

    Args:
        doc_id: The document identifier to delete.
    """
    collection = _get_collection()
    collection.delete(where={"doc_id": doc_id})
    logger.info("Deleted all chunks for doc_id='%s'.", doc_id)


def get_document_chunks(doc_id: str) -> list[dict]:
    """Return all chunks for a specific document, ordered by chunk_index.

    Args:
        doc_id: The document identifier.

    Returns:
        list[dict]: Chunk dicts with "text" and "metadata" keys, sorted by
        chunk_index. Empty list if the document is not found.
    """
    collection = _get_collection()

    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(results.get("documents", []), results.get("metadatas", [])):
        chunks.append({"text": doc, "metadata": meta})

    chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    return chunks


def list_documents() -> list[dict]:
    """List all unique documents stored in ChromaDB.

    Reconstructs the document list by grouping chunks on their ``doc_id``
    metadata field.

    Returns:
        list[dict]: One entry per document: {id, filename, upload_date, chunks}.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    docs: dict[str, dict] = {}
    chunk_counts: dict[str, int] = defaultdict(int)

    for meta in metadatas:
        doc_id = meta.get("doc_id", "")
        if doc_id not in docs:
            docs[doc_id] = {
                "id": doc_id,
                "filename": meta.get("filename", ""),
                "upload_date": meta.get("upload_date", ""),
            }
        chunk_counts[doc_id] += 1

    return [
        {**doc_info, "chunks": chunk_counts[doc_id]}
        for doc_id, doc_info in docs.items()
    ]
