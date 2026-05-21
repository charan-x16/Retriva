"""FastAPI entrypoint for PDF ingestion and citation-grounded querying."""

from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.db.qdrant_client import init_qdrant, upsert_chunks
from backend.generation.answer_gen import generate_answer
from backend.generation.llm_provider import get_llm_provider
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.parsers import detect_and_parse
from backend.reranker.bge_reranker import load_reranker, rerank
from backend.retrieval.bm25_retriever import embed_sparse_texts, retrieve_bm25
from backend.retrieval.dense_retriever import (
    embed_texts,
    load_embedding_model,
    retrieve_dense,
)
from backend.retrieval.fusion import reciprocal_rank_fusion

load_dotenv()

app = FastAPI(title="Retriva", version="0.1.0")

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
def query_docs(request: QueryRequest):
    """Answer a question with hybrid retrieval, reranking, and citations."""

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    if _is_greeting(question):
        return {
            "answer": (
                "Hello. Upload and ingest a PDF, then ask me anything about it."
            ),
            "citations": [],
            "chunks": [],
        }

    client = get_qdrant_client()
    model = get_embedding_model()

    bm25_results = retrieve_bm25(client, question, top_k=20)
    dense_results = retrieve_dense(client, model, question, top_k=20)
    fused_chunks = reciprocal_rank_fusion([bm25_results, dense_results])
    if not fused_chunks:
        raise HTTPException(status_code=404, detail="No indexed chunks found.")

    top_chunks = rerank(get_bge_reranker(), question, fused_chunks, top_k=5)
    answer_payload = generate_answer(get_llm(), question, top_chunks)

    return {
        "answer": answer_payload["answer"],
        "citations": answer_payload["citations"],
        "chunks": top_chunks,
    }


def _is_greeting(text: str) -> bool:
    """Return whether the user message is a simple greeting."""

    normalized = text.lower().strip(" .,!?\n\t")
    return normalized in GREETING_WORDS
