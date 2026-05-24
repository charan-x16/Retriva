"""ColPali visual page indexing and retrieval with Qdrant multivectors."""

import os
import re
import uuid
from pathlib import Path

import fitz
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

from backend.pdf_utils import quiet_mupdf
from backend.visual.page_renderer import render_page_as_image

quiet_mupdf()

COLPALI_VECTOR_NAME = "colpali"
COLPALI_VECTOR_SIZE = 128


def load_colpali_model():
    """Load the configured ColPali model and processor."""

    from colpali_engine.models import ColPali, ColPaliProcessor

    load_dotenv()
    model_name = os.getenv("COLPALI_MODEL", "vidore/colpali-v1.2")
    cache_dir = os.getenv("MODEL_CACHE_DIR", ".cache/models")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    try:
        model = ColPali.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            dtype=torch_dtype,
        ).to(device)
    except TypeError:
        model = ColPali.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
        ).to(device)
    model.eval()
    try:
        processor = ColPaliProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=False,
        )
    except TypeError:
        processor = ColPaliProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    return model, processor


def index_pdf_pages(model, processor, pdf_path, collection_name) -> None:
    """Render PDF pages, embed them with ColPali, and store page vectors."""

    load_dotenv()
    client = _init_visual_qdrant(collection_name)
    source = os.path.basename(pdf_path)
    batch_size = _visual_upsert_batch_size()

    with fitz.open(pdf_path) as document:
        page_count = document.page_count
        page_texts = [
            document.load_page(index).get_text("text").strip()
            for index in range(page_count)
        ]

    points = []
    for page_number in range(1, page_count + 1):
        image = render_page_as_image(pdf_path, page_number, dpi=150)
        image_path = _save_page_image(image, source, page_number)
        page_embedding = _encode_page(model, processor, image)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{page_number}"))
        points.append(
            PointStruct(
                id=point_id,
                vector={COLPALI_VECTOR_NAME: page_embedding},
                payload={
                    "page": page_number,
                    "source": source,
                    "text": page_texts[page_number - 1],
                    "image_path": image_path,
                    "type": "visual_page",
                },
            )
        )
        if len(points) >= batch_size:
            _upsert_visual_points(client, collection_name, points)
            points = []

    if points:
        _upsert_visual_points(client, collection_name, points)


def retrieve_visual(model, processor, query, collection_name, top_k=5) -> list[dict]:
    """Retrieve top matching visual PDF pages for a text query."""

    if not query.strip():
        return []

    client = _init_visual_qdrant(collection_name)
    query_embedding = _encode_query(model, processor, query)
    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        using=COLPALI_VECTOR_NAME,
        limit=top_k,
        with_payload=True,
        timeout=_visual_timeout(),
    )

    results = []
    for point in response.points:
        payload = dict(point.payload or {})
        results.append(
            {
                "page": payload.get("page"),
                "source": payload.get("source"),
                "text": payload.get("text", ""),
                "image_path": payload.get("image_path"),
                "score": float(point.score),
            }
        )
    return results


def get_visual_collection_name() -> str:
    """Return the configured Qdrant collection for ColPali page vectors."""

    load_dotenv()
    return os.getenv("COLPALI_COLLECTION_NAME", "retriva_visual_pages")


def visual_index_has_points(collection_name=None) -> bool:
    """Return whether the visual collection exists and contains page vectors."""

    load_dotenv()
    collection_name = collection_name or get_visual_collection_name()
    client = _visual_qdrant_client()
    if not _collection_exists(client, collection_name):
        return False

    try:
        result = client.count(collection_name=collection_name, exact=False)
        return int(result.count) > 0
    except Exception:
        return False


def list_visual_documents(collection_name=None) -> list[dict]:
    """Return unique document names and visual page counts from Qdrant."""

    load_dotenv()
    collection_name = collection_name or get_visual_collection_name()
    client = _visual_qdrant_client()
    if not _collection_exists(client, collection_name):
        return []

    documents = {}
    try:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                source = payload.get("source")
                if not source:
                    continue

                item = documents.setdefault(
                    source,
                    {
                        "source": source,
                        "visual_pages": set(),
                    },
                )
                if payload.get("page"):
                    item["visual_pages"].add(payload["page"])

            if offset is None:
                break
    except Exception:
        return []

    results = []
    for item in documents.values():
        results.append(
            {
                "source": item["source"],
                "visual_pages": len(item["visual_pages"]),
            }
        )
    return sorted(results, key=lambda item: item["source"].lower())


def _init_visual_qdrant(collection_name) -> QdrantClient:
    """Connect to Qdrant and ensure the ColPali collection exists."""

    client = _visual_qdrant_client()

    if not _collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                COLPALI_VECTOR_NAME: VectorParams(
                    size=COLPALI_VECTOR_SIZE,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(m=0),
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )
    else:
        _ensure_visual_schema(client, collection_name)
    return client


def _visual_qdrant_client() -> QdrantClient:
    """Create a Qdrant client for visual index operations."""

    load_dotenv()
    qdrant_url = (
        os.getenv("QDRANT_URL")
        or os.getenv("QDRANT_ENDPOINT")
        or "http://localhost:6333"
    )
    client = QdrantClient(
        url=qdrant_url,
        api_key=os.getenv("QDRANT_API_KEY") or None,
        timeout=_visual_timeout(),
    )
    return client


def _upsert_visual_points(client, collection_name, points) -> None:
    """Upload ColPali page vectors to Qdrant in small timeout-safe batches."""

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
        timeout=_visual_timeout(),
    )


def _visual_upsert_batch_size() -> int:
    """Return how many visual page points to upload per Qdrant request."""

    return max(1, int(os.getenv("COLPALI_UPSERT_BATCH_SIZE", "1")))


def _visual_timeout() -> int:
    """Return the Qdrant timeout for large visual vector operations."""

    return int(os.getenv("QDRANT_VISUAL_TIMEOUT", "300"))


def _save_page_image(image, source, page_number) -> str:
    """Persist a rendered PDF page for multimodal answer generation."""

    image_dir = _visual_page_dir() / _safe_name(Path(source).stem)
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"page-{page_number}.jpg"
    image.save(image_path, format="JPEG", quality=85, optimize=True)
    return str(image_path)


def _visual_page_dir() -> Path:
    """Return the local directory used for rendered visual pages."""

    return Path(os.getenv("VISUAL_PAGE_IMAGE_DIR", "storage/visual_pages"))


def _safe_name(value) -> str:
    """Make a filesystem-safe folder name from a document source name."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return safe or "document"


def _encode_page(model, processor, image) -> list[list[float]]:
    """Encode a page image into ColPali multivectors."""

    device = _model_device(model)
    batch = processor.process_images([image]).to(device)
    with torch.no_grad():
        embeddings = model(**batch)
    return embeddings[0].float().cpu().tolist()


def _encode_query(model, processor, query) -> list[list[float]]:
    """Encode a text query into ColPali multivectors."""

    device = _model_device(model)
    batch = processor.process_queries([query]).to(device)
    with torch.no_grad():
        embeddings = model(**batch)
    return embeddings[0].float().cpu().tolist()


def _model_device(model) -> str:
    """Return the active device for a ColPali model."""

    return str(next(model.parameters()).device)


def _collection_exists(client, collection_name) -> bool:
    """Return whether a Qdrant collection exists."""

    if hasattr(client, "collection_exists"):
        return client.collection_exists(collection_name)

    try:
        client.get_collection(collection_name)
        return True
    except Exception:
        return False


def _ensure_visual_schema(client, collection_name) -> None:
    """Validate that an existing collection supports ColPali multivectors."""

    collection = client.get_collection(collection_name)
    vectors = getattr(collection.config.params, "vectors", None)
    if not isinstance(vectors, dict) or COLPALI_VECTOR_NAME not in vectors:
        raise ValueError(
            f"Collection '{collection_name}' must contain named vector "
            f"'{COLPALI_VECTOR_NAME}'."
        )

    vector_config = vectors[COLPALI_VECTOR_NAME]
    if getattr(vector_config, "size", None) != COLPALI_VECTOR_SIZE:
        raise ValueError(
            f"Collection '{collection_name}' must use vector size "
            f"{COLPALI_VECTOR_SIZE} for ColPali."
        )
