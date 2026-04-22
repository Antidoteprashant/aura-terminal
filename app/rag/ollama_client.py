"""HTTP client for the Ollama local LLM server."""

import json
import logging
from typing import Generator

import requests

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT = 120  # seconds — Gemma 4 on CPU is slow


def _base_url() -> str:
    """Return the Ollama base URL from Flask config or fall back to env/default."""
    try:
        from flask import current_app
        return current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    except RuntimeError:
        import os
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def check_connection() -> bool:
    """Ping the Ollama server to verify it is running.

    Returns:
        bool: True if reachable, False otherwise.
    """
    try:
        resp = requests.get(f"{_base_url()}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def generate_stream(prompt: str, model: str) -> Generator[str, None, None]:
    """Stream token-by-token output from Ollama.

    POSTs to /api/generate with stream=true and yields each token as it
    arrives. On connection errors, yields a user-friendly error message
    instead of raising.

    Args:
        prompt: The full prompt string to send to the model.
        model: The Ollama model name (e.g. "gemma4-e2b").

    Yields:
        str: Individual tokens from the model response.
    """
    url = f"{_base_url()}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": True}

    try:
        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("response", "")
                if token:
                    yield token

                if chunk.get("done", False):
                    break

    except requests.exceptions.ConnectionError:
        logger.error("Ollama is not running at %s", _base_url())
        yield "\n\n[ERROR: Cannot connect to Ollama. Is it running? Try: ollama serve]"
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out after %ds", OLLAMA_TIMEOUT)
        yield "\n\n[ERROR: Ollama request timed out. The model may be overloaded.]"
    except requests.exceptions.HTTPError as exc:
        logger.error("Ollama HTTP error: %s", exc)
        yield f"\n\n[ERROR: Ollama returned an error — {exc}]"
    except Exception as exc:
        logger.error("Unexpected Ollama error: %s", exc)
        yield f"\n\n[ERROR: Unexpected error communicating with Ollama — {exc}]"


def generate(prompt: str, model: str) -> str:
    """Send a prompt to Ollama and return the full response (non-streaming).

    Used for quick internal tasks like query rewriting where streaming
    is unnecessary overhead.

    Args:
        prompt: The full prompt string to send to the model.
        model: The Ollama model name (e.g. "gemma4-e2b").

    Returns:
        str: The complete response text, or an error message.
    """
    url = f"{_base_url()}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as exc:
        logger.error("Ollama non-streaming error: %s", exc)
        return ""
    except Exception as exc:
        logger.error("Unexpected Ollama error (non-stream): %s", exc)
        return ""


def list_models() -> list[str]:
    """Return the names of all models available in the local Ollama instance.

    Returns:
        list[str]: Model name strings, empty list on error.
    """
    try:
        resp = requests.get(f"{_base_url()}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.exceptions.RequestException:
        return []
