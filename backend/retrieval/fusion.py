"""Reciprocal Rank Fusion for combining ranked retrieval results."""


def reciprocal_rank_fusion(results_list: list[list], k=60) -> list[dict]:
    """Fuse multiple ranked result lists and deduplicate by chunk_id."""

    scores = {}
    chunks_by_id = {}

    for results in results_list:
        for rank, item in enumerate(results, start=1):
            chunk = _result_to_chunk(item)
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunks_by_id:
                chunks_by_id[chunk_id] = chunk
            else:
                chunks_by_id[chunk_id].update(
                    {
                        key: value
                        for key, value in chunk.items()
                        if key.endswith("_score")
                    }
                )

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    fused = []
    for chunk_id in ranked_ids:
        chunk = dict(chunks_by_id[chunk_id])
        chunk["rrf_score"] = scores[chunk_id]
        fused.append(chunk)
    return fused


def _result_to_chunk(item) -> dict:
    """Normalize dense chunks and BM25 (chunk, score) tuples."""

    if isinstance(item, tuple) and len(item) == 2:
        chunk, score = item
        normalized = dict(chunk)
        normalized["bm25_score"] = float(score)
        return normalized
    return dict(item)

