"""Self-correcting retrieval pipeline with grade and rewrite steps."""

from backend.correction.grader import grade_context
from backend.correction.rewriter import rewrite_query
from backend.reranker.bge_reranker import rerank
from backend.retrieval.bm25_retriever import retrieve_bm25
from backend.retrieval.dense_retriever import retrieve_dense
from backend.retrieval.fusion import reciprocal_rank_fusion

GRADE_THRESHOLD = 0.7


def self_correcting_retrieve(
    llm,
    bm25_index,
    bm25_chunks,
    qdrant_client,
    embed_model,
    reranker,
    query,
    top_k=5,
) -> dict:
    """Retrieve, grade context, and rewrite/retrieve again when quality is weak."""

    chunks = _retrieve_once(qdrant_client, embed_model, reranker, query, top_k)
    grade_score = grade_context(llm, query, chunks)

    if grade_score < GRADE_THRESHOLD:
        new_query = rewrite_query(llm, query)
        if new_query and new_query != query:
            corrected_chunks = _retrieve_once(
                qdrant_client,
                embed_model,
                reranker,
                new_query,
                top_k,
            )
            return {
                "chunks": corrected_chunks,
                "query_used": new_query,
                "was_corrected": True,
                "grade_score": grade_score,
            }

    return {
        "chunks": chunks,
        "query_used": query,
        "was_corrected": False,
        "grade_score": grade_score,
    }


def _retrieve_once(qdrant_client, embed_model, reranker, query, top_k) -> list[dict]:
    """Run BM25 + dense retrieval, RRF fusion, and reranking once."""

    bm25_results = retrieve_bm25(qdrant_client, query, top_k=20)
    dense_results = retrieve_dense(qdrant_client, embed_model, query, top_k=20)
    fused_chunks = reciprocal_rank_fusion([bm25_results, dense_results])
    if not fused_chunks:
        return []
    return rerank(reranker, query, fused_chunks, top_k=top_k)

