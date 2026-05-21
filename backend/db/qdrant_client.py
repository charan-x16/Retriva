"""Qdrant setup, upsert, dense search, and sparse BM25-style search helpers."""

import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)


def init_qdrant() -> QdrantClient:
    """Connect to Qdrant and create the chunk collection if needed."""

    load_dotenv()
    qdrant_url = (
        os.getenv("QDRANT_URL")
        or os.getenv("QDRANT_ENDPOINT")
        or os.getenv("QDARANT_ENDPOINT")
        or "http://localhost:6333"
    )
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or None
    collection_name = get_collection_name()
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        cloud_inference=should_use_cloud_inference(qdrant_api_key),
    )

    if _collection_exists(client, collection_name):
        if should_recreate_collection():
            client.delete_collection(collection_name=collection_name)
            _create_collection(client, collection_name)
            return client
        _ensure_collection_schema(client, collection_name)
    else:
        _create_collection(client, collection_name)
    return client


def should_recreate_collection() -> bool:
    """Return whether startup may recreate the configured Qdrant collection."""

    load_dotenv()
    return os.getenv("QDRANT_RECREATE_COLLECTION", "false").lower() == "true"


def should_use_cloud_inference(qdrant_api_key=None) -> bool:
    """Return whether Qdrant should use cloud inference for BM25 vectors."""

    load_dotenv()
    default = "true" if qdrant_api_key else "false"
    return os.getenv("QDRANT_CLOUD_INFERENCE", default).lower() == "true"


def get_dense_vector_name() -> str:
    """Return the configured dense vector name in Qdrant."""

    load_dotenv()
    return os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")


def get_sparse_vector_name() -> str:
    """Return the configured sparse BM25 vector name in Qdrant."""

    load_dotenv()
    return os.getenv("QDRANT_SPARSE_VECTOR_NAME", "bm25")


def get_vector_size() -> int:
    """Return the configured embedding vector size."""

    load_dotenv()
    return int(os.getenv("EMBEDDING_VECTOR_SIZE", "768"))


def get_collection_name() -> str:
    """Return the configured Qdrant collection name."""

    load_dotenv()
    return os.getenv("QDRANT_COLLECTION_NAME", "retriva_chunks")


def upsert_chunks(client, chunks_with_embeddings):
    """Store chunk vectors and chunk payloads in Qdrant."""

    collection_name = get_collection_name()
    dense_vector_name = get_dense_vector_name()
    sparse_vector_name = get_sparse_vector_name()
    points = []
    for chunk in chunks_with_embeddings:
        embedding = chunk["embedding"]
        sparse_embedding = chunk["sparse_embedding"]
        payload = {
            key: value
            for key, value in chunk.items()
            if key not in {"embedding", "sparse_embedding"}
        }
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, payload["chunk_id"]))
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    dense_vector_name: embedding,
                    sparse_vector_name: sparse_embedding,
                },
                payload=payload,
            )
        )

    if points:
        client.upsert(collection_name=collection_name, points=points)


def search_dense(client, query_embedding, top_k=20) -> list[dict]:
    """Return nearest chunks from Qdrant for a query embedding."""

    collection_name = get_collection_name()
    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using=get_dense_vector_name(),
        limit=top_k,
        with_payload=True,
    )
    return _points_to_chunks(response.points, "dense_score")


def search_sparse(client, query_sparse_vector, top_k=20) -> list[dict]:
    """Return nearest chunks from Qdrant sparse BM25-style search."""

    response = client.query_points(
        collection_name=get_collection_name(),
        query=query_sparse_vector,
        using=get_sparse_vector_name(),
        limit=top_k,
        with_payload=True,
    )
    return _points_to_chunks(response.points, "bm25_score")


def _collection_exists(client, collection_name) -> bool:
    """Check collection existence across qdrant-client versions."""

    if hasattr(client, "collection_exists"):
        return client.collection_exists(collection_name)

    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def _create_collection(client, collection_name):
    """Create the hybrid dense plus sparse Qdrant collection."""

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            get_dense_vector_name(): VectorParams(
                size=get_vector_size(),
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            get_sparse_vector_name(): SparseVectorParams(modifier=Modifier.IDF)
        },
    )


def _ensure_collection_schema(client, collection_name):
    """Fail fast if an existing collection cannot support hybrid retrieval."""

    collection = client.get_collection(collection_name)
    params = collection.config.params
    dense_vectors = getattr(params, "vectors", None)
    sparse_vectors = getattr(params, "sparse_vectors", None)
    dense_vector_name = get_dense_vector_name()
    sparse_vector_name = get_sparse_vector_name()

    if not isinstance(dense_vectors, dict) or dense_vector_name not in dense_vectors:
        raise ValueError(
            f"Collection '{collection_name}' must contain dense vector "
            f"'{dense_vector_name}'. Create a new collection or recreate this one."
        )

    dense_config = dense_vectors[dense_vector_name]
    if getattr(dense_config, "size", None) != get_vector_size():
        raise ValueError(
            f"Collection '{collection_name}' vector size does not match "
            "EMBEDDING_VECTOR_SIZE."
        )

    if not isinstance(sparse_vectors, dict) or sparse_vector_name not in sparse_vectors:
        client.update_collection(
            collection_name=collection_name,
            sparse_vectors_config={
                sparse_vector_name: SparseVectorParams(modifier=Modifier.IDF)
            },
        )


def _points_to_chunks(points, score_name) -> list[dict]:
    """Convert Qdrant scored points to chunk dictionaries."""

    chunks = []
    for point in points:
        payload = dict(point.payload or {})
        payload[score_name] = float(point.score)
        chunks.append(payload)
    return chunks
