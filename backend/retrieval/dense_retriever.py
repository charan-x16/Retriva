"""Dense retrieval with BGE embeddings and Qdrant vector search."""

from sentence_transformers import SentenceTransformer

from backend.db.qdrant_client import search_dense


def load_embedding_model():
    """Load the BGE base embedding model."""

    return SentenceTransformer("BAAI/bge-base-en-v1.5")


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Embed texts and return JSON-serializable float vectors."""

    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def retrieve_dense(client, model, query, top_k=20) -> list[dict]:
    """Embed a query and retrieve nearest chunks from Qdrant."""

    query_embedding = embed_texts(model, [query])[0]
    return search_dense(client, query_embedding, top_k=top_k)

