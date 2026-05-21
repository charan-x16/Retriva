# Retriva

Retriva is a Stage 1 self-correcting hybrid document RAG pipeline for PDFs.

## What is included

- FastAPI backend with `/ingest` and `/query`
- PDF text extraction, OCR fallback, and Camelot table extraction
- Character chunking with overlap
- Qdrant sparse BM25-style retrieval, dense BGE retrieval, RRF fusion, and BGE reranking
- OpenAI/OpenRouter answer generation with inline source citations
- Streamlit frontend for upload and question answering
- Qdrant-backed vector storage through Docker Compose

## Setup

Copy `.env.example` to `.env` and set your Qdrant and LLM keys.

For Qdrant Cloud, set `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION_NAME`.
The backend also accepts `QDRANT_ENDPOINT` and the current `QDARANT_ENDPOINT` spelling.

Set `EMBEDDING_MODEL` and `EMBEDDING_VECTOR_SIZE` together. The default is
`BAAI/bge-base-en-v1.5` with vector size `768`.
Set `MODEL_CACHE_DIR=.cache/models` to keep downloaded embedding and reranker
weights in a stable local cache across backend restarts.

Retriva stores dense vectors and sparse BM25-style vectors in the same Qdrant
collection. If you already created a collection with the old single-vector
schema, use a new `QDRANT_COLLECTION_NAME` or set
`QDRANT_RECREATE_COLLECTION=true` once to recreate the existing collection.
The sparse side uses Qdrant's `qdrant/bm25` model by default.

For OpenRouter, set `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, and `OPENROUTER_MODEL`.
If `LLM_PROVIDER` is omitted but `OPENROUTER_API_KEY` exists, Retriva uses OpenRouter.

```powershell
pip install -r requirements.txt
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

The OCR and table paths also need Tesseract OCR and Ghostscript installed on the host.
