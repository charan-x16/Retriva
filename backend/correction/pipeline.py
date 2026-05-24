"""Self-correcting retrieval pipeline with grade and rewrite steps."""

import logging
import os
from contextlib import contextmanager
from time import perf_counter

from backend.correction.grader import grade_context
from backend.correction.rewriter import rewrite_query
from backend.reranker.bge_reranker import rerank
from backend.retrieval.bm25_retriever import retrieve_bm25
from backend.retrieval.dense_retriever import retrieve_dense
from backend.retrieval.fusion import reciprocal_rank_fusion

GRADE_THRESHOLD = 0.7
logger = logging.getLogger("uvicorn.error")


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

    chunks = _timed_call(
        "retrieval.original",
        _retrieve_once,
        qdrant_client,
        embed_model,
        reranker,
        query,
        top_k,
    )
    if not _crag_enabled():
        return {
            "chunks": chunks,
            "query_used": query,
            "was_corrected": False,
            "grade_score": None,
        }

    grade_score = _timed_call(
        "retrieval.grade_context",
        grade_context,
        llm,
        query,
        chunks,
    )

    if grade_score < _grade_threshold() and _query_rewrite_enabled():
        new_query = _timed_call("retrieval.rewrite_query", rewrite_query, llm, query)
        if new_query and new_query != query:
            corrected_chunks = _timed_call(
                "retrieval.corrected",
                _retrieve_once,
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

    bm25_results = _timed_call(
        "retrieval.bm25",
        retrieve_bm25,
        qdrant_client,
        query,
        top_k=20,
    )
    dense_results = _timed_call(
        "retrieval.dense",
        retrieve_dense,
        qdrant_client,
        embed_model,
        query,
        top_k=20,
    )
    fused_chunks = _timed_call(
        "retrieval.rrf",
        reciprocal_rank_fusion,
        [bm25_results, dense_results],
    )
    if not fused_chunks:
        return []
    return _timed_call(
        "retrieval.rerank",
        rerank,
        reranker,
        query,
        fused_chunks,
        top_k=top_k,
    )


def _crag_enabled() -> bool:
    """Return whether CRAG grading is enabled."""

    return os.getenv("ENABLE_CRAG", "true").lower() == "true"


def _query_rewrite_enabled() -> bool:
    """Return whether low-grade queries may be rewritten."""

    return os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"


def _grade_threshold() -> float:
    """Return the CRAG grade threshold."""

    return float(os.getenv("CRAG_GRADE_THRESHOLD", str(GRADE_THRESHOLD)))


@contextmanager
def _timed_stage(label):
    """Log elapsed time for one correction stage."""

    started_at = perf_counter()
    try:
        yield
    finally:
        logger.info("retriva timing | %s | %.2fs", label, perf_counter() - started_at)


def _timed_call(label, func, *args, **kwargs):
    """Run a callable and log elapsed time."""

    with _timed_stage(label):
        return func(*args, **kwargs)
