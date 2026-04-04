"""Entry point for the RAG Terminal application.

Environment detection
---------------------
Local  (default) : Ollama LLM  + ChromaDB vector store
Hosted (RENDER=1 or PRODUCTION=1) : Groq via langchain_groq + Pinecone

Required env vars in hosted mode
---------------------------------
  GROQ_API_KEY     — Groq API key (https://console.groq.com)
  PINECONE_API_KEY — Pinecone API key (https://app.pinecone.io)
  PINECONE_INDEX   — Pinecone index name
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Environment detection ────────────────────────────────────────────────────

IS_HOSTED = bool(os.environ.get("RENDER") or os.environ.get("PRODUCTION"))

if IS_HOSTED:
    logger.info("Hosted environment detected.")

    # ── Validate required keys ───────────────────────────────────────────────
    missing: list[str] = []

    groq_key     = os.environ.get("GROQ_API_KEY", "").strip()
    pinecone_key = os.environ.get("PINECONE_API_KEY", "").strip()
    pinecone_idx = os.environ.get("PINECONE_INDEX", "").strip()

    if not groq_key:
        missing.append(
            "  GROQ_API_KEY   — get a free key at https://console.groq.com"
        )
    if not pinecone_key:
        missing.append(
            "  PINECONE_API_KEY — get a key at https://app.pinecone.io"
        )
    if not pinecone_idx:
        missing.append(
            "  PINECONE_INDEX   — the name of your Pinecone index"
        )

    if missing:
        sys.exit(
            "\n[FATAL] Hosted environment detected but required API keys are missing.\n"
            "Set the following environment variables before starting the app:\n\n"
            + "\n".join(missing)
            + "\n\nFor local development unset RENDER and PRODUCTION."
        )

    logger.info("All hosted API keys present — using Groq + Pinecone.")
    os.environ["IS_HOSTED"] = "1"          # propagate so Config picks it up

else:
    logger.info("Local environment — using Ollama + ChromaDB.")
    os.environ.pop("IS_HOSTED", None)


# ── Create and run the app ───────────────────────────────────────────────────

from app import create_app  # noqa: E402 — must come after env setup

app = create_app()

if __name__ == "__main__":
    app.run(
        host=app.config.get("FLASK_HOST", "0.0.0.0"),
        port=int(app.config.get("FLASK_PORT", 5000)),
        debug=app.config.get("FLASK_DEBUG", False) and not IS_HOSTED,
    )
