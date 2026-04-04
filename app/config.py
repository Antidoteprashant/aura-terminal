"""Configuration loaded from environment variables via python-dotenv."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration with defaults for every value."""

    # Environment mode
    # Set to True automatically by run.py when RENDER or PRODUCTION is detected.
    IS_HOSTED: bool = os.getenv("IS_HOSTED", "").lower() in ("1", "true", "yes")

    # Ollama (local LLM — used when IS_HOSTED is False)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4-e2b")

    # Groq (hosted LLM — used when IS_HOSTED is True, or as Ollama fallback locally)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Pinecone (hosted vector store — used when IS_HOSTED is True)
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "rag-terminal")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "documents")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # Flask
    FLASK_HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", 5000))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    # Upload
    UPLOAD_FOLDER: str = os.getenv("UPLOAD_FOLDER", "./uploads")
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", 16777216))
    ALLOWED_EXTENSIONS: set = {
        ext.strip().lower()
        for ext in os.getenv("ALLOWED_EXTENSIONS", "pdf,txt,md").split(",")
        if ext.strip()
    }
