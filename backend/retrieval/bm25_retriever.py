"""Qdrant-backed BM25 retriever for document chunks."""

import os

from dotenv import load_dotenv
from qdrant_client.models import Document

from backend.db.qdrant_client import search_sparse


def build_bm25_index(chunks):
    """No-op compatibility hook; Qdrant builds the sparse index on upsert."""

    return None, chunks


def embed_sparse_texts(texts: list[str]) -> list[Document]:
    """Create Qdrant BM25 document vectors for indexing."""

    return [embed_sparse_text(text) for text in texts]


def embed_sparse_text(text: str) -> Document:
    """Create one Qdrant BM25 document vector."""

    return Document(text=text, model=get_bm25_model())


def retrieve_bm25(client, query, top_k=20) -> list[dict]:
    """Return top chunks from Qdrant BM25 search."""

    query_vector = embed_sparse_text(query)
    if not query.strip():
        return []
    return search_sparse(client, query_vector, top_k=top_k)


def get_bm25_model() -> str:
    """Return the configured Qdrant BM25 model name."""

    load_dotenv()
    return os.getenv("BM25_MODEL", "qdrant/bm25")
