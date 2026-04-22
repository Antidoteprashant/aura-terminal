"""Groq API client — OpenAI-compatible streaming, used as Ollama fallback."""

import json
import logging
from typing import Generator

import requests

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TIMEOUT  = 60  # seconds


def _api_key() -> str:
    try:
        from flask import current_app
        return current_app.config.get("GROQ_API_KEY", "")
    except RuntimeError:
        import os
        return os.getenv("GROQ_API_KEY", "")


def has_api_key() -> bool:
    return bool(_api_key())


def check_connection() -> bool:
    """Return True if the Groq API key is set and the endpoint is reachable."""
    key = _api_key()
    if not key:
        return False
    try:
        resp = requests.get(
            f"{GROQ_BASE_URL}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def generate_stream(prompt: str, model: str) -> Generator[str, None, None]:
    """Stream tokens from Groq using the OpenAI-compatible chat completions API.

    Args:
        prompt: The full prompt string.
        model:  Groq model ID (e.g. "llama-3.1-8b-instant").

    Yields:
        str: Individual content tokens.
    """
    key = _api_key()
    if not key:
        yield "[ERROR: GROQ_API_KEY is not set in .env. Add it to use Groq as fallback.]"
        return

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "stream":      True,
        "temperature": 0.7,
    }

    try:
        with requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=GROQ_TIMEOUT,
        ) as resp:
            if resp.status_code == 401:
                yield "[ERROR: Invalid Groq API key. Check GROQ_API_KEY in .env]"
                return
            if resp.status_code == 429:
                yield "[ERROR: Groq rate limit exceeded. Try again in a moment.]"
                return
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                content = (
                    chunk.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if content:
                    yield content

    except requests.exceptions.ConnectionError:
        yield "[ERROR: Cannot reach Groq API. Check your internet connection.]"
    except requests.exceptions.Timeout:
        yield "[ERROR: Groq request timed out.]"
    except requests.exceptions.HTTPError as exc:
        yield f"[ERROR: Groq API error — {exc}]"
    except Exception as exc:
        logger.error("Unexpected Groq error: %s", exc)
        yield f"[ERROR: Unexpected Groq error — {exc}]"


def generate(prompt: str, model: str) -> str:
    """Send a prompt to Groq and return the full response (non-streaming).

    Used for quick internal tasks like query rewriting.

    Args:
        prompt: The full prompt string.
        model:  Groq model ID (e.g. "llama-3.1-8b-instant").

    Returns:
        str: The complete response text, or empty string on error.
    """
    key = _api_key()
    if not key:
        return ""

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "stream":      False,
        "temperature": 0.3,   # lower temp for deterministic rewrites
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=GROQ_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.error("Groq non-streaming error: %s", exc)
        return ""
