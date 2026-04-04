"""File parsing for PDF, TXT, and Markdown documents."""

import logging
import os
import re

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def parse_file(filepath: str) -> dict:
    """Parse a file and return its text content with metadata.

    Supports PDF (via PyPDF2), TXT (UTF-8), and Markdown (markdown-it-py).
    Raises ValueError for unsupported file types or unreadable files.
    Returns empty text (with a logged warning) for files with no extractable text.

    Args:
        filepath: Absolute or relative path to the file.

    Returns:
        dict: {"text": str, "metadata": dict}
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        text = _parse_pdf(filepath)
    elif ext == ".txt":
        text = _parse_txt(filepath)
    else:  # .md
        text = _parse_markdown(filepath)

    if not text.strip():
        logger.warning("File '%s' produced empty text after parsing.", filepath)

    return {
        "text": text,
        "metadata": {
            "filename": os.path.basename(filepath),
            "extension": ext,
            "filepath": filepath,
        },
    }


def _parse_pdf(filepath: str) -> str:
    """Extract text from a PDF file page by page using PyPDF2.

    Args:
        filepath: Path to the PDF file.

    Returns:
        str: Extracted text joined with newlines.

    Raises:
        ValueError: If the PDF is corrupted, encrypted, or unreadable.
    """
    import PyPDF2

    try:
        pages = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            if reader.is_encrypted:
                raise ValueError(
                    "PDF is encrypted/password-protected and cannot be read."
                )

            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)

        return "\n".join(pages)

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF: {exc}") from exc


def _parse_txt(filepath: str) -> str:
    """Read a plain-text file with UTF-8 encoding.

    Encoding errors are handled gracefully via 'replace' error mode.

    Args:
        filepath: Path to the TXT file.

    Returns:
        str: File contents.

    Raises:
        ValueError: If the file cannot be read.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as exc:
        raise ValueError(f"Could not read text file: {exc}") from exc


def _parse_markdown(filepath: str) -> str:
    """Strip Markdown formatting and return plain text using markdown-it-py.

    Renders the Markdown to HTML then strips all HTML tags so only the
    readable prose remains.

    Args:
        filepath: Path to the Markdown file.

    Returns:
        str: Plain text extracted from Markdown.

    Raises:
        ValueError: If the file cannot be read.
    """
    from markdown_it import MarkdownIt

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
    except OSError as exc:
        raise ValueError(f"Could not read Markdown file: {exc}") from exc

    md = MarkdownIt()
    html = md.render(raw)
    # Strip HTML tags; collapse whitespace
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain
