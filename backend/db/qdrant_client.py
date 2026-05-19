"""Qdrant setup, upsert, and dense vector search helpers."""

import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "retriva_chunks"
VECTOR_SIZE = 768


def init_qdrant() -> QdrantClient:
    """Connect to Qdrant and create the chunk collection if needed."""

    load_dotenv()
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=qdrant_url)

    if not _collection_exists(client, COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    return client


def upsert_chunks(client, chunks_with_embeddings):
    """Store chunk vectors and chunk payloads in Qdrant."""

    points = []
    for chunk in chunks_with_embeddings:
        embedding = chunk["embedding"]
        payload = {key: value for key, value in chunk.items() if key != "embedding"}
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, payload["chunk_id"]))
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)


def search_dense(client, query_embedding, top_k=20) -> list[dict]:
    """Return nearest chunks from Qdrant for a query embedding."""

    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )
    except AttributeError:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        results = response.points

    chunks = []
    for result in results:
        payload = dict(result.payload or {})
        payload["dense_score"] = float(result.score)
        chunks.append(payload)
    return chunks


def _collection_exists(client, collection_name) -> bool:
    """Check collection existence across qdrant-client versions."""

    if hasattr(client, "collection_exists"):
        return client.collection_exists(collection_name)

    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False

