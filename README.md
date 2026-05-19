# Retriva

Retriva is a Stage 1 self-correcting hybrid document RAG pipeline for PDFs.

## What is included

- FastAPI backend with `/ingest` and `/query`
- PDF text extraction, OCR fallback, and Camelot table extraction
- Character chunking with overlap
- BM25 retrieval, dense BGE retrieval, RRF fusion, and BGE reranking
- OpenAI answer generation with inline source citations
- Streamlit frontend for upload and question answering
- Qdrant-backed vector storage through Docker Compose

## Setup

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

For Docker Compose, keep `QDRANT_URL=http://qdrant:6333`.
For local backend runs outside Docker, set `QDRANT_URL=http://localhost:6333`.

```powershell
pip install -r requirements.txt
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

The OCR and table paths also need Tesseract OCR and Ghostscript installed on the host.
