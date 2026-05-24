"""BGE reranker loading and scoring for candidate chunks."""

import os

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker


def load_reranker() -> FlagReranker:
    """Load the configured BGE reranker model."""

    load_dotenv()
    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    cache_dir = os.getenv("MODEL_CACHE_DIR", ".cache/models")
    use_fp16 = os.getenv("RERANKER_USE_FP16", "false").lower() == "true"
    return FlagReranker(model_name, use_fp16=use_fp16, cache_dir=cache_dir)


def rerank(reranker, query, chunks, top_k=5) -> list[dict]:
    """Score query-chunk pairs and return the best chunks."""

    if not chunks:
        return []

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.compute_score(pairs)
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if not isinstance(scores, list):
        scores = [scores]

    ranked = []
    for chunk, score in zip(chunks, scores):
        item = dict(chunk)
        item["rerank_score"] = float(score)
        ranked.append(item)

    return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)[:top_k]
