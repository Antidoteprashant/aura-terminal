"""Embedding generation using sentence-transformers (singleton pattern)."""

import logging

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused for the lifetime of the process
_model = None


def _get_model():
    """Load the embedding model on first call and cache it.

    Reads the model name from app config (EMBEDDING_MODEL). Falls back to
    all-MiniLM-L6-v2 if config is unavailable (e.g. during tests).

    Returns:
        SentenceTransformer: The loaded model instance.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            from flask import current_app
            model_name = current_app.config.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        except RuntimeError:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    return _model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Uses all-MiniLM-L6-v2 which produces 384-dimensional vectors.
    Processes texts in a single batch for efficiency.

    Args:
        texts: List of text strings to embed.

    Returns:
        list[list[float]]: One embedding vector per input text.
    """
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [v.tolist() for v in vectors]
