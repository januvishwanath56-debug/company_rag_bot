# src/retriever.py
# Handles storing chunks into ChromaDB and retrieving relevant chunks for a query

import chromadb
from src.embeddings import embed_texts, embed_query

CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "support_docs"

_client     = None
_collection = None

def get_collection():
    """Returns the ChromaDB collection, connecting only once."""
    global _client, _collection
    if _collection is None:
        _client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection

def store_chunks(chunks: list):
    """
    Embeds and stores all chunks into ChromaDB.
    Deletes any existing collection first so re-running is safe.
    """
    global _client, _collection

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete old collection if exists
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    embeddings = embed_texts(chunks)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    # Reset cached references so next get_collection() picks up fresh data
    _collection = None
    _client     = None

    return len(chunks)

def retrieve_chunks(query: str, top_k: int = 3) -> list:
    """
    Embeds the query and retrieves the top_k most similar chunks from ChromaDB.
    Returns a list of text strings.
    """
    collection     = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results["documents"][0] if results["documents"] else []
