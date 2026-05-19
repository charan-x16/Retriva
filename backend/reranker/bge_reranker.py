"""BGE reranker loading and scoring for candidate chunks."""

from FlagEmbedding import FlagReranker


def load_reranker() -> FlagReranker:
    """Load the BGE reranker model."""

    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)


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

