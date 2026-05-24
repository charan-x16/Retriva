"""FastAPI entrypoint for automatic document ingestion and grounded querying."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from tempfile import TemporaryDirectory

import fitz
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.correction.pipeline import self_correcting_retrieve
from backend.db.qdrant_client import (
    init_qdrant,
    list_indexed_documents,
    upsert_chunks,
)
from backend.evaluation.logger import (
    get_all_logs,
    init_db,
    log_query,
    update_ragas_scores,
)
from backend.evaluation.ragas_eval import compute_ragas
from backend.generation.answer_gen import generate_answer
from backend.generation.llm_provider import get_llm_provider
from backend.generation.vision_answer_gen import generate_visual_answer
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.pdf_analyzer import analyze_pdf, should_index_visual
from backend.ingestion.parsers import detect_and_parse
from backend.pdf_utils import quiet_mupdf, repair_pdf
from backend.reranker.bge_reranker import load_reranker
from backend.retrieval.bm25_retriever import embed_sparse_texts
from backend.retrieval.dense_retriever import (
    embed_texts,
    load_embedding_model,
)
from backend.visual.colpali_retriever import (
    get_visual_collection_name,
    index_pdf_pages,
    list_visual_documents,
    load_colpali_model,
    retrieve_visual,
    visual_index_has_points,
)

load_dotenv()
quiet_mupdf()

app = FastAPI(title="Retriva", version="0.1.0")
init_db()
logger = logging.getLogger("uvicorn.error")

_qdrant_client = None
_embedding_model = None
_reranker = None
_llm = None
_colpali_model = None
_colpali_processor = None

GREETING_WORDS = {
    "hello",
    "hi",
    "hey",
    "yo",
    "namaste",
    "good morning",
    "good afternoon",
    "good evening",
}

RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


class QueryRequest(BaseModel):
    """Request body for document question answering."""

    question: str
    mode: str | None = None


def get_qdrant_client():
    """Return a cached Qdrant client."""

    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = init_qdrant()
    return _qdrant_client


def get_embedding_model():
    """Return a cached embedding model."""

    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_embedding_model()
    return _embedding_model


def get_bge_reranker():
    """Return a cached BGE reranker."""

    global _reranker
    if _reranker is None:
        _reranker = load_reranker()
    return _reranker


def get_llm():
    """Return a cached LLM provider."""

    global _llm
    if _llm is None:
        _llm = get_llm_provider()
    return _llm


def get_colpali():
    """Return a cached ColPali model and processor."""

    global _colpali_model, _colpali_processor
    if _colpali_model is None or _colpali_processor is None:
        _colpali_model, _colpali_processor = load_colpali_model()
    return _colpali_model, _colpali_processor


@contextmanager
def timed_stage(label):
    """Log how long one backend stage takes."""

    started_at = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - started_at
        logger.info("retriva timing | %s | %.2fs", label, elapsed)


def timed_call(label, func, *args, **kwargs):
    """Run a callable and log its elapsed time."""

    with timed_stage(label):
        return func(*args, **kwargs)


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest one PDF into text and, when useful, visual indexes."""

    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / filename
        file_path.write_bytes(await file.read())
        ingest_path = timed_call(
            "ingest.prepare_pdf",
            _prepare_pdf_for_ingestion,
            file_path,
        )
        analysis = timed_call("ingest.analyze_pdf", analyze_pdf, ingest_path)

        text_result = timed_call(
            "ingest.text_index",
            _ingest_text_index,
            ingest_path,
            filename,
        )
        should_visual, visual_reason = should_index_visual(analysis, text_result)
        if should_visual:
            visual_result = timed_call(
                "ingest.visual_index",
                _ingest_visual_index,
                ingest_path,
                filename,
            )
            visual_result["decision_reason"] = visual_reason
        else:
            visual_result = {
                "status": "skipped",
                "source": filename,
                "pages": analysis.get("pages", 0),
                "reason": visual_reason,
            }

    if text_result["status"] == "failed" and visual_result["status"] != "indexed":
        raise HTTPException(
            status_code=400,
            detail="Could not index this PDF with either text or visual retrieval.",
        )

    return {
        "status": "ok",
        "source": filename,
        "text": text_result,
        "visual": visual_result,
        "analysis": _public_pdf_analysis(analysis),
    }


@app.post("/ingest_visual")
async def ingest_visual_pdf(file: UploadFile = File(...)):
    """Ingest one PDF into the ColPali visual page index."""

    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    collection_name = get_visual_collection_name()
    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / filename
        file_path.write_bytes(await file.read())
        ingest_path = _prepare_pdf_for_ingestion(file_path)

        pages = _count_pdf_pages(ingest_path)
        if pages < 1:
            raise HTTPException(status_code=400, detail="PDF has no pages.")

        result = _ingest_visual_index(ingest_path, filename)
        if result["status"] == "failed":
            raise HTTPException(status_code=500, detail=result["reason"])

    return {
        "status": "indexed",
        "source": filename,
        "pages": pages,
        "collection": collection_name,
    }


@app.post("/query")
def query_docs(request: QueryRequest, background_tasks: BackgroundTasks):
    """Answer a question with hybrid retrieval, reranking, and citations."""

    total_started_at = perf_counter()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    if _is_greeting(question):
        answer = "Hello. Upload and ingest a PDF, then ask me anything about it."
        log_query(question, answer, [], None)
        return {
            "answer": answer,
            "citations": [],
            "chunks": [],
            "was_corrected": False,
            "grade_score": None,
            "original_query": question,
            "query_used": question,
        }

    if _is_library_question(question):
        return _library_answer_response(question)

    llm = timed_call("query.load_llm_provider", get_llm)

    text_result = timed_call(
        "query.text_retrieval",
        _try_retrieve_text_evidence,
        llm,
        question,
        top_k=5,
    )
    visual_results = []
    if _should_retrieve_visual(text_result) and _visual_fallback_enabled():
        visual_results = timed_call(
            "query.visual_retrieval",
            _try_retrieve_visual_evidence,
            question,
            top_k=5,
        )
    answer_payload = timed_call(
        "query.answer_generation",
        _generate_auto_answer,
        llm,
        question,
        text_result,
        visual_results,
    )
    if not answer_payload.get("answer"):
        raise HTTPException(status_code=404, detail="No indexed evidence found.")

    contexts = answer_payload.get("contexts", [])
    row_id = log_query(
        question=question,
        answer=answer_payload["answer"],
        contexts=contexts,
        grade_score=text_result.get("grade_score"),
    )
    if contexts:
        background_tasks.add_task(
            _compute_and_store_ragas,
            row_id,
            question,
            answer_payload["answer"],
            contexts,
            get_embedding_model(),
        )

    response = {
        "answer": answer_payload["answer"],
        "citations": answer_payload["citations"],
        "chunks": text_result.get("chunks", []),
        "visual_results": visual_results,
        "answer_mode": answer_payload.get("answer_mode"),
        "retrieval_summary": _build_retrieval_summary(text_result, visual_results),
        "was_corrected": text_result.get("was_corrected", False),
        "grade_score": text_result.get("grade_score"),
        "original_query": question,
        "query_used": text_result.get("query_used", question),
    }
    logger.info(
        "retriva timing | query.total | %.2fs",
        perf_counter() - total_started_at,
    )
    return response


@app.post("/query_visual")
def query_visual_docs(request: QueryRequest):
    """Return text and ColPali visual retrieval results for comparison."""

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    llm = get_llm()
    text_result = {"chunks": []}
    if request.mode != "Visual (ColPali)":
        text_result = _try_retrieve_text_evidence(llm, question, top_k=5)

    colpali_model, colpali_processor = get_colpali()
    visual_results = retrieve_visual(
        colpali_model,
        colpali_processor,
        question,
        get_visual_collection_name(),
        top_k=5,
    )
    text_results = text_result["chunks"]
    visual_answer = _generate_visual_answer_with_fallback(
        question,
        visual_results,
        llm,
    )

    return {
        "answer": visual_answer["answer"],
        "citations": visual_answer["citations"],
        "answer_mode": visual_answer.get("answer_mode", "visual_multimodal"),
        "text_results": text_results,
        "visual_results": visual_results,
        "comparison_note": _build_comparison_note(text_results, visual_results),
    }


@app.get("/eval_logs")
def eval_logs():
    """Return all logged query evaluations."""

    return get_all_logs()


@app.get("/documents")
def documents():
    """Return document names and counts currently stored in Qdrant."""

    merged = _current_documents()
    return {
        "total": len(merged),
        "documents": merged,
    }


@app.post("/eval_logs/recompute")
def recompute_eval_logs(background_tasks: BackgroundTasks):
    """Re-run Ragas for logged rows that still have pending scores."""

    logs = get_all_logs()
    pending_logs = [
        row
        for row in logs
        if row.get("contexts")
        and any(row.get(metric) is None for metric in RAGAS_METRICS)
    ]

    model = get_embedding_model()
    for row in pending_logs:
        background_tasks.add_task(
            _compute_and_store_ragas,
            row["id"],
            row["question"],
            row["answer"],
            row["contexts"],
            model,
        )

    return {
        "status": "queued",
        "rows": len(pending_logs),
    }


def _is_greeting(text: str) -> bool:
    """Return whether the user message is a simple greeting."""

    normalized = text.lower().strip(" .,!?\n\t")
    return normalized in GREETING_WORDS


def _is_library_question(text: str) -> bool:
    """Return whether the user is asking what documents are indexed."""

    normalized = " ".join(text.lower().strip(" .,!?\n\t").split())
    inventory_terms = (
        "which docs",
        "what docs",
        "which documents",
        "what documents",
        "which files",
        "what files",
        "docs do you have",
        "documents do you have",
        "files do you have",
        "show library",
        "list docs",
        "list documents",
        "list files",
        "stored in qdrant",
    )
    return any(term in normalized for term in inventory_terms)


def _library_answer_response(question: str) -> dict:
    """Build a direct answer from the indexed document library."""

    docs = _current_documents()
    if not docs:
        answer = "No documents are currently indexed in Qdrant."
    else:
        names = ", ".join(doc["source"] for doc in docs)
        answer = f"I currently have {len(docs)} document(s): {names}."

    log_query(question, answer, [], None)
    return {
        "answer": answer,
        "citations": [],
        "chunks": [],
        "visual_results": [],
        "answer_mode": "library_inventory",
        "retrieval_summary": {
            "text_chunks": sum(doc.get("text_chunks", 0) for doc in docs),
            "visual_pages": sum(doc.get("visual_pages", 0) for doc in docs),
            "document_count": len(docs),
        },
        "was_corrected": False,
        "grade_score": None,
        "original_query": question,
        "query_used": question,
    }


def _current_documents() -> list[dict]:
    """Return merged text and visual document summaries from Qdrant."""

    text_documents = list_indexed_documents(get_qdrant_client())
    visual_documents = list_visual_documents(get_visual_collection_name())
    return _merge_document_indexes(text_documents, visual_documents)


def _prepare_pdf_for_ingestion(file_path: Path) -> str:
    """Return a repaired PDF path when PyMuPDF can clean the upload."""

    try:
        return repair_pdf(file_path, file_path)
    except Exception:
        return str(file_path)


def _public_pdf_analysis(analysis) -> dict:
    """Return compact PDF analysis details for the ingest response."""

    return {
        "pages": analysis.get("pages", 0),
        "avg_text_chars_per_page": analysis.get("avg_text_chars_per_page", 0),
        "low_text_pages": analysis.get("low_text_pages", 0),
        "meaningful_image_pages": analysis.get("meaningful_image_pages", 0),
        "scanned_like_pages": analysis.get("scanned_like_pages", 0),
        "drawing_heavy_pages": analysis.get("drawing_heavy_pages", 0),
    }


def _merge_document_indexes(text_documents, visual_documents) -> list[dict]:
    """Merge text and visual Qdrant document summaries by filename."""

    merged = {}
    for item in text_documents:
        source = item["source"]
        merged[source] = {
            "source": source,
            "text_chunks": item.get("text_chunks", 0),
            "text_pages": item.get("text_pages", 0),
            "visual_pages": 0,
            "has_text": item.get("text_chunks", 0) > 0,
            "has_visual": False,
        }

    for item in visual_documents:
        source = item["source"]
        entry = merged.setdefault(
            source,
            {
                "source": source,
                "text_chunks": 0,
                "text_pages": 0,
                "visual_pages": 0,
                "has_text": False,
                "has_visual": False,
            },
        )
        entry["visual_pages"] = item.get("visual_pages", 0)
        entry["has_visual"] = item.get("visual_pages", 0) > 0

    return sorted(merged.values(), key=lambda item: item["source"].lower())


def _ingest_text_index(file_path: str, filename: str) -> dict:
    """Extract PDF text/tables, chunk it, and index text vectors in Qdrant."""

    try:
        docs = timed_call("ingest.text_parse", detect_and_parse, file_path)
        chunks = timed_call("ingest.text_chunk", chunk_documents, docs)
        if not chunks:
            return {
                "status": "failed",
                "source": filename,
                "documents": len(docs),
                "chunks": 0,
                "reason": "No text could be extracted.",
            }

        model = timed_call("ingest.load_embedding_model", get_embedding_model)
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = timed_call(
            "ingest.embed_dense_chunks",
            embed_texts,
            model,
            chunk_texts,
        )
        sparse_embeddings = timed_call(
            "ingest.embed_sparse_chunks",
            embed_sparse_texts,
            chunk_texts,
        )
        chunks_with_embeddings = [
            {**chunk, "embedding": embedding, "sparse_embedding": sparse_embedding}
            for chunk, embedding, sparse_embedding in zip(
                chunks,
                embeddings,
                sparse_embeddings,
            )
        ]

        client = timed_call("ingest.load_qdrant_client", get_qdrant_client)
        timed_call(
            "ingest.upsert_text_chunks",
            upsert_chunks,
            client,
            chunks_with_embeddings,
        )
        return {
            "status": "indexed",
            "source": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "source": filename,
            "documents": 0,
            "chunks": 0,
            "reason": str(exc),
        }


def _ingest_visual_index(file_path: str, filename: str) -> dict:
    """Render PDF pages, embed them with ColPali, and index visual evidence."""

    try:
        pages = _count_pdf_pages(file_path)
        if pages < 1:
            return {
                "status": "failed",
                "source": filename,
                "pages": 0,
                "reason": "PDF has no pages.",
            }

        collection_name = get_visual_collection_name()
        model, processor = timed_call("ingest.load_colpali", get_colpali)
        timed_call(
            "ingest.index_visual_pages",
            index_pdf_pages,
            model,
            processor,
            file_path,
            collection_name,
        )
        return {
            "status": "indexed",
            "source": filename,
            "pages": pages,
            "collection": collection_name,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "source": filename,
            "pages": 0,
            "reason": str(exc),
        }


def _retrieve_text_evidence(llm, model, question, top_k=5) -> dict:
    """Run the existing self-correcting text retrieval path."""

    return self_correcting_retrieve(
        llm=llm,
        bm25_index=None,
        bm25_chunks=None,
        qdrant_client=get_qdrant_client(),
        embed_model=model,
        reranker=get_bge_reranker(),
        query=question,
        top_k=top_k,
    )


def _try_retrieve_text_evidence(llm, question, top_k=5) -> dict:
    """Return text evidence if available, otherwise keep visual retrieval alive."""

    try:
        model = get_embedding_model()
        return _retrieve_text_evidence(llm, model, question, top_k=top_k)
    except Exception:
        return {"chunks": [], "grade_score": None, "was_corrected": False}


def _try_retrieve_visual_evidence(question, top_k=5) -> list[dict]:
    """Return ColPali page evidence if the visual index is available."""

    try:
        collection_name = get_visual_collection_name()
        if not visual_index_has_points(collection_name):
            return []
        model, processor = get_colpali()
        return retrieve_visual(
            model,
            processor,
            question,
            collection_name,
            top_k=top_k,
        )
    except Exception:
        return []


def _should_retrieve_visual(text_result) -> bool:
    """Return whether text evidence is weak enough to need visual retrieval."""

    chunks = text_result.get("chunks", [])
    grade_score = text_result.get("grade_score")
    if not chunks:
        return True
    if grade_score is None:
        return False
    return grade_score < _auto_text_grade_threshold()


def _visual_fallback_enabled() -> bool:
    """Return whether query-time visual fallback is allowed."""

    if os.getenv("VISUAL_INDEX_MODE", "auto").lower() == "never":
        return False
    return os.getenv("ENABLE_VISUAL_FALLBACK", "true").lower() == "true"


def _generate_auto_answer(llm, question, text_result, visual_results) -> dict:
    """Choose the strongest available evidence path and generate one answer."""

    text_chunks = text_result.get("chunks", [])
    grade_score = text_result.get("grade_score")
    text_is_strong = (
        bool(text_chunks)
        and grade_score is not None
        and grade_score >= _auto_text_grade_threshold()
    )

    if text_is_strong or (text_chunks and not visual_results):
        return _generate_text_answer(llm, question, text_chunks, "text_hybrid")

    visual_payload = None
    if visual_results:
        visual_payload = _generate_visual_answer_with_fallback(
            question,
            visual_results,
            llm,
        )
        visual_payload["contexts"] = _visual_contexts(visual_results)
        if visual_payload.get("answer_mode") in {
            "visual_multimodal",
            "generation_rate_limited",
            "generation_error",
        }:
            return visual_payload
        if not text_chunks:
            return visual_payload

    if text_chunks:
        return _generate_text_answer(llm, question, text_chunks, "text_fallback")

    if visual_payload:
        return visual_payload

    return {
        "answer": "",
        "citations": [],
        "contexts": [],
        "answer_mode": "none",
    }


def _generate_text_answer(llm, question, chunks, answer_mode) -> dict:
    """Generate an answer from text chunks and attach eval contexts."""

    contexts = [chunk.get("text", "") for chunk in chunks]
    try:
        answer_payload = generate_answer(llm, question, chunks)
        answer_payload["answer_mode"] = answer_mode
        answer_payload["contexts"] = contexts
        return answer_payload
    except Exception as exc:
        return _generation_error_payload(exc, contexts)


def _auto_text_grade_threshold() -> float:
    """Return the grade above which text evidence is trusted directly."""

    return float(os.getenv("AUTO_TEXT_GRADE_THRESHOLD", "0.7"))


def _visual_contexts(visual_results) -> list[str]:
    """Return OCR text from visual pages for logging and best-effort eval."""

    return [result.get("text", "") for result in visual_results if result.get("text")]


def _build_retrieval_summary(text_result, visual_results) -> dict:
    """Summarize automatic retrieval evidence counts for the client."""

    return {
        "text_chunks": len(text_result.get("chunks", [])),
        "visual_pages": len(visual_results),
        "text_grade": text_result.get("grade_score"),
        "text_query_used": text_result.get("query_used"),
    }


def _build_comparison_note(text_results, visual_results) -> str:
    """Summarize how text and visual retrieval compare for one query."""

    base_note = (
        f"Text pipeline found {len(text_results)} chunks. "
        f"ColPali found {len(visual_results)} pages."
    )
    text_pages = {result.get("page") for result in text_results if result.get("page")}
    visual_pages = {
        result.get("page") for result in visual_results if result.get("page")
    }
    overlap = sorted(text_pages & visual_pages)

    if overlap:
        pages = ", ".join(str(page) for page in overlap[:3])
        return f"{base_note} Both modes agree on page(s): {pages}."
    if text_results and visual_results:
        return (
            f"{base_note} They surfaced different pages, so compare exact OCR "
            "text against visual layout evidence."
        )
    if visual_results:
        return f"{base_note} Only ColPali returned visual page candidates."
    if text_results:
        return f"{base_note} Only the OCR text pipeline returned chunks."
    return f"{base_note} No relevant evidence was found by either mode."


def _visual_results_to_chunks(visual_results) -> list[dict]:
    """Convert visual page hits into answer-generation context chunks."""

    chunks = []
    for index, result in enumerate(visual_results, start=1):
        chunks.append(
            {
                "text": result.get("text") or "",
                "chunk_id": (
                    f"visual:{result.get('source', 'unknown')}:"
                    f"{result.get('page', index)}"
                ),
                "parent_id": (
                    f"{result.get('source', 'unknown')}:"
                    f"{result.get('page', index)}"
                ),
                "page": result.get("page", index),
                "source": result.get("source", "unknown"),
                "chunk_index": index - 1,
                "visual_score": result.get("score"),
            }
        )
    return chunks


def _generate_visual_answer_with_fallback(question, visual_results, llm) -> dict:
    """Generate from page images, with text fallback if the vision call fails."""

    try:
        return generate_visual_answer(question, visual_results)
    except Exception:
        visual_chunks = _visual_results_to_chunks(visual_results)
        try:
            answer_payload = generate_answer(llm, question, visual_chunks)
            answer_payload["answer_mode"] = "text_fallback"
            answer_payload["contexts"] = [
                chunk.get("text", "") for chunk in visual_chunks
            ]
            return answer_payload
        except Exception as exc:
            return _generation_error_payload(
                exc,
                [chunk.get("text", "") for chunk in visual_chunks],
            )


def _generation_error_payload(exc, contexts) -> dict:
    """Return a user-safe answer when the LLM provider fails."""

    if _is_rate_limit_error(exc):
        return {
            "answer": (
                "The relevant evidence was retrieved, but the configured "
                "OpenRouter model is temporarily rate-limited upstream. Retry "
                "shortly, or switch OPENROUTER_MODEL/VISUAL_OPENROUTER_MODEL "
                "to another available model."
            ),
            "citations": [],
            "contexts": contexts,
            "answer_mode": "generation_rate_limited",
        }

    return {
        "answer": (
            "The relevant evidence was retrieved, but answer generation failed. "
            "Please check the backend logs for the LLM provider error."
        ),
        "citations": [],
        "contexts": contexts,
        "answer_mode": "generation_error",
    }


def _is_rate_limit_error(exc) -> bool:
    """Return whether an exception looks like an LLM provider rate limit."""

    message = str(exc).lower()
    return (
        exc.__class__.__name__ == "RateLimitError"
        or "rate limit" in message
        or "rate-limited" in message
        or "error code: 429" in message
    )


def _count_pdf_pages(file_path: str) -> int:
    """Return the number of pages in a PDF."""

    with fitz.open(file_path) as document:
        return document.page_count


def _compute_and_store_ragas(row_id, question, answer, contexts, embed_model) -> None:
    """Compute Ragas scores in the background and update the log row."""

    scores = compute_ragas(question, answer, contexts, embed_model=embed_model)
    update_ragas_scores(row_id, scores)
