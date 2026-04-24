# src/embeddings.py
# Handles loading the embedding model and generating embeddings

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, ~90MB, downloads once automatically

_model = None  # global so we only load it once per session

def get_embedding_model():
    """Returns the embedding model, loading it only once."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: list) -> list:
    """
    Takes a list of strings and returns a list of embedding vectors.
    Each vector is a list of floats (384 dimensions for MiniLM).
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.tolist() for e in embeddings]

def embed_query(query: str) -> list:
    """
    Embeds a single query string.
    Returns a single embedding vector as a list of floats.
    """
    model = get_embedding_model()
    return model.encode([query])[0].tolist()
