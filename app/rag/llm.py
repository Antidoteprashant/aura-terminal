"""Unified LLM router.

Local  (IS_HOSTED=False) : Ollama → Groq fallback (plain requests)
Hosted (IS_HOSTED=True)  : Groq via langchain_groq.ChatGroq (no Ollama check)

The resolve_backend result is cached for 30 seconds to avoid a health-check
HTTP round-trip on every token-stream request in local mode.
"""

import logging
import time
from typing import Generator

logger = logging.getLogger(__name__)

# ── Backend cache ─────────────────────────────────────────────────────────────
_cache: dict = {"backend": None, "model": "", "ts": 0.0}
_CACHE_TTL = 30  # seconds


def _is_hosted() -> bool:
    try:
        from flask import current_app
        return current_app.config.get("IS_HOSTED", False)
    except RuntimeError:
        import os
        return os.getenv("IS_HOSTED", "").lower() in ("1", "true", "yes")


def _ollama_model() -> str:
    try:
        from flask import current_app
        return current_app.config.get("OLLAMA_MODEL", "gemma4-e2b")
    except RuntimeError:
        import os
        return os.getenv("OLLAMA_MODEL", "gemma4-e2b")


def _groq_model() -> str:
    try:
        from flask import current_app
        return current_app.config.get("GROQ_MODEL", "llama-3.1-8b-instant")
    except RuntimeError:
        import os
        return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def _groq_api_key() -> str:
    try:
        from flask import current_app
        return current_app.config.get("GROQ_API_KEY", "")
    except RuntimeError:
        import os
        return os.getenv("GROQ_API_KEY", "")


def resolve_backend() -> tuple[str, str]:
    """Return (backend, model). Uses cache to avoid repeated health-checks.

    Hosted mode  → always "groq" (langchain_groq path)
    Local mode   → tries Ollama, falls back to plain Groq client, then "none"

    Returns:
        tuple[str, str]: (backend_name, model_name)
            backend_name: "ollama" | "groq" | "groq_langchain" | "none"
    """
    now = time.time()
    if _cache["backend"] is not None and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["backend"], _cache["model"]

    if _is_hosted():
        # Hosted: always use langchain_groq — key was validated at startup
        result = ("groq_langchain", _groq_model())
        logger.info("LLM backend: langchain_groq (%s) [hosted]", result[1])
    else:
        # Local: try Ollama first, then plain Groq client
        from app.rag.ollama_client import check_connection as ollama_up
        if ollama_up():
            result = ("ollama", _ollama_model())
            logger.info("LLM backend: Ollama (%s)", result[1])
        else:
            from app.rag.groq_client import has_api_key
            if has_api_key():
                result = ("groq", _groq_model())
                logger.info("LLM backend: Groq (%s) — Ollama unavailable", result[1])
            else:
                result = ("none", "")
                logger.warning(
                    "LLM backend: none — Ollama offline and no GROQ_API_KEY set"
                )

    _cache["backend"] = result[0]
    _cache["model"]   = result[1]
    _cache["ts"]      = now
    return result


def invalidate_cache() -> None:
    """Force re-detection on the next call (e.g. after config change)."""
    _cache["backend"] = None


def generate_stream(prompt: str) -> Generator[str, None, None]:
    """Route a prompt to the active backend and stream tokens.

    Args:
        prompt: The fully-formatted prompt string.

    Yields:
        str: Individual tokens from the LLM.
    """
    backend, model = resolve_backend()

    if backend == "ollama":
        from app.rag.ollama_client import generate_stream as _stream
        yield from _stream(prompt, model)

    elif backend == "groq_langchain":
        yield from _groq_langchain_stream(prompt, model)

    elif backend == "groq":
        from app.rag.groq_client import generate_stream as _stream
        yield from _stream(prompt, model)

    else:
        yield (
            "[ERROR: No LLM available. "
            "Either start Ollama (ollama serve) or add GROQ_API_KEY to .env]"
        )


def _groq_langchain_stream(prompt: str, model: str) -> Generator[str, None, None]:
    """Stream tokens using langchain_groq.ChatGroq.

    Args:
        prompt: The fully-formatted prompt string.
        model:  Groq model ID (e.g. "llama-3.1-8b-instant").

    Yields:
        str: Individual content tokens.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
    except ImportError:
        yield (
            "[ERROR: langchain_groq is not installed. "
            "Run: pip install langchain-groq]"
        )
        return

    api_key = _groq_api_key()
    if not api_key:
        yield "[ERROR: GROQ_API_KEY is not set. Cannot use hosted LLM.]"
        return

    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name=model,
            streaming=True,
            temperature=0.7,
        )
        for chunk in llm.stream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield chunk.content

    except Exception as exc:
        logger.error("langchain_groq stream error: %s", exc)
        yield f"[ERROR: Groq LLM error — {exc}]"
