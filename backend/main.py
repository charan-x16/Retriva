"""FastAPI entrypoint for PDF ingestion and citation-grounded querying."""

from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.correction.pipeline import self_correcting_retrieve
from backend.db.qdrant_client import init_qdrant, upsert_chunks
from backend.evaluation.logger import (
    get_all_logs,
    init_db,
    log_query,
    update_ragas_scores,
)
from backend.evaluation.ragas_eval import compute_ragas
from backend.generation.answer_gen import generate_answer
from backend.generation.llm_provider import get_llm_provider
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.parsers import detect_and_parse
from backend.reranker.bge_reranker import load_reranker
from backend.retrieval.bm25_retriever import embed_sparse_texts
from backend.retrieval.dense_retriever import (
    embed_texts,
    load_embedding_model,
)

load_dotenv()

app = FastAPI(title="Retriva", version="0.1.0")
init_db()

_qdrant_client = None
_embedding_model = None
_reranker = None
_llm = None

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


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """Ingest one PDF and index dense plus sparse vectors in Qdrant."""

    filename = Path(file.filename or "upload.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / filename
        file_path.write_bytes(await file.read())

        docs = detect_and_parse(str(file_path))
        chunks = chunk_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted.")

        model = get_embedding_model()
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(model, chunk_texts)
        sparse_embeddings = embed_sparse_texts(chunk_texts)
        chunks_with_embeddings = [
            {**chunk, "embedding": embedding, "sparse_embedding": sparse_embedding}
            for chunk, embedding, sparse_embedding in zip(
                chunks,
                embeddings,
                sparse_embeddings,
            )
        ]

        client = get_qdrant_client()
        upsert_chunks(client, chunks_with_embeddings)

    return {
        "status": "ok",
        "source": filename,
        "documents": len(docs),
        "chunks": len(chunks),
    }


@app.post("/query")
def query_docs(request: QueryRequest, background_tasks: BackgroundTasks):
    """Answer a question with hybrid retrieval, reranking, and citations."""

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

    client = get_qdrant_client()
    model = get_embedding_model()
    llm = get_llm()

    retrieval_result = self_correcting_retrieve(
        llm=llm,
        bm25_index=None,
        bm25_chunks=None,
        qdrant_client=client,
        embed_model=model,
        reranker=get_bge_reranker(),
        query=question,
        top_k=5,
    )
    top_chunks = retrieval_result["chunks"]
    if not top_chunks:
        raise HTTPException(status_code=404, detail="No indexed chunks found.")

    answer_payload = generate_answer(llm, question, top_chunks)
    contexts = [chunk.get("text", "") for chunk in top_chunks]
    row_id = log_query(
        question=question,
        answer=answer_payload["answer"],
        contexts=contexts,
        grade_score=retrieval_result["grade_score"],
    )
    background_tasks.add_task(
        _compute_and_store_ragas,
        row_id,
        question,
        answer_payload["answer"],
        contexts,
        model,
    )

    return {
        "answer": answer_payload["answer"],
        "citations": answer_payload["citations"],
        "chunks": top_chunks,
        "was_corrected": retrieval_result["was_corrected"],
        "grade_score": retrieval_result["grade_score"],
        "original_query": question,
        "query_used": retrieval_result["query_used"],
    }


@app.get("/eval_logs")
def eval_logs():
    """Return all logged query evaluations."""

    return get_all_logs()


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


def _compute_and_store_ragas(row_id, question, answer, contexts, embed_model) -> None:
    """Compute Ragas scores in the background and update the log row."""

    scores = compute_ragas(question, answer, contexts, embed_model=embed_model)
    update_ragas_scores(row_id, scores)
