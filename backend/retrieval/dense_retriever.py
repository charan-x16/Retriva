"""Dense retrieval with configurable embeddings and Qdrant vector search."""

import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from backend.db.qdrant_client import search_dense


def load_embedding_model():
    """Load the configured embedding model."""

    load_dotenv()
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    cache_dir = os.getenv("MODEL_CACHE_DIR", ".cache/models")
    return SentenceTransformer(model_name, cache_folder=cache_dir)


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Embed texts and return JSON-serializable float vectors."""

    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def retrieve_dense(client, model, query, top_k=20) -> list[dict]:
    """Embed a query and retrieve nearest chunks from Qdrant."""

    query_embedding = embed_texts(model, [query])[0]
    return search_dense(client, query_embedding, top_k=top_k)
