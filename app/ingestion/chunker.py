"""Text chunking using LangChain's RecursiveCharacterTextSplitter."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[dict]:
    """Split text into overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter. If the text is shorter than
    chunk_size, it is returned as a single chunk.

    Args:
        text: The raw document text to split.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        list[dict]: Each item is {"text": str, "chunk_index": int, "start_char": int}
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    docs = splitter.create_documents([text])

    return [
        {
            "text": doc.page_content,
            "chunk_index": i,
            "start_char": doc.metadata.get("start_index", 0),
        }
        for i, doc in enumerate(docs)
    ]
