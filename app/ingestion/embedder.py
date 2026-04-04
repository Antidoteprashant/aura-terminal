"""Hybrid Embedding generation (Hugging Face API for Cloud, Local for Dev)."""

import os
import logging
import requests

logger = logging.getLogger(__name__)

# Module-level singleton for local mode
_model = None

# Free Hugging Face Inference API endpoint for your exact model
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def _get_local_model():
    """Load the local PyTorch model only if we aren't using the cloud API."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local all-MiniLM-L6-v2 model into memory...")
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed locally. "
                "Run `pip install sentence-transformers` if you want to run this locally without an API key."
            )
    return _model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings using a smart Hybrid approach:
    1. If HF_TOKEN is in environment -> Route to Hugging Face API (Render Mode)
    2. If no token -> Load model into local memory (MacBook Mode)
    """
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        # ==========================================
        # CLOUD MODE (Render / Production)
        # ==========================================
        logger.info("HF_TOKEN found. Routing embeddings to Hugging Face API...")
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            # The API returns a list of lists of floats, exactly what we need
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error("Hugging Face API Failed: %s", e)
            if 'response' in locals() and response is not None:
                logger.error("API Response: %s", response.text)
            raise RuntimeError("Cloud embedding generation failed.")

    else:
        # ==========================================
        # LOCAL MODE (Development)
        # ==========================================
        logger.info("No HF_TOKEN found. Processing embeddings locally...")
        model = _get_local_model()
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]