# src/chunker.py
# Splits a large text into overlapping chunks for embedding

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Splits text into chunks of chunk_size characters,
    with overlap between consecutive chunks so context isn't lost.
    Returns a list of text strings.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
