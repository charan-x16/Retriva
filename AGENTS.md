# Agent Guide for Retriva

This file gives coding agents the repo-specific context needed to work on
Retriva without rediscovering the system from scratch.

## Project Summary

Retriva is a self-correcting hybrid document RAG system for PDFs.

It ingests normal text PDFs, scanned PDFs, tables, forms, charts, and visual
page layouts. It retrieves evidence from Qdrant with dense BGE vectors, sparse
BM25 vectors, RRF fusion, BGE reranking, CRAG-style correction, and optional
ColPali visual page retrieval. Answers are generated through the configured LLM
provider with grounded citations.

The user experience should stay simple:

```text
Upload PDF -> Ingest -> Ask question -> Receive cited answer -> Inspect details if needed
```

## Golden Rules

- Keep Python simple and readable.
- Keep new code close to the existing module boundaries.
- Every new Python file should start with a short docstring.
- Put prompts in `backend/generation/prompt.py`, not inline in pipeline files.
- Use `.env` for runtime config. Never hard-code secrets, keys, model names, or
  Qdrant URLs in code.
- Do not print or expose real API keys from `.env`.
- Do not remove the reranker. It is an important quality layer. If latency is a
  problem, use a smaller reranker model through `.env`.
- Do not make the user choose text vs visual retrieval in normal chat. The app
  should automatically use visual retrieval when the document needs it.
- Keep large generated data out of git: model cache, Qdrant data, SQLite logs,
  rendered page images, and `__pycache__`.

## Main Commands

Install dependencies:

```powershell
uv pip install -r requirements.txt
```

Run the backend:

```powershell
uvicorn backend.main:app
```

Run the backend while editing backend files:

```powershell
uvicorn backend.main:app --reload --reload-dir backend
```

Do not use plain `uvicorn backend.main:app --reload` while editing frontend
files. It watches the whole repo, restarts the backend, and reloads large models.

Run the frontend:

```powershell
streamlit run frontend/app.py
```

Check indexed documents:

```powershell
Invoke-RestMethod http://localhost:8000/documents
```

Quick syntax check:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m compileall backend frontend
```

## Current Architecture

### Backend

`backend/main.py` is the FastAPI entry point and orchestration layer.

Important endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /ingest` | Upload a PDF and index text evidence plus visual evidence when needed. |
| `POST /query` | Ask a question and return one cited answer from the best evidence path. |
| `GET /documents` | List indexed documents from Qdrant. |
| `GET /eval_logs` | Read evaluation rows from SQLite. |
| `POST /eval_logs/recompute` | Queue recomputation for pending evaluation scores. |
| `POST /ingest_visual` | Debug endpoint for visual-only indexing. |
| `POST /query_visual` | Debug endpoint for visual comparison retrieval. |

Core folders:

| Folder | Responsibility |
| --- | --- |
| `backend/ingestion/` | PDF analysis, parsing, OCR, table extraction, and chunking. |
| `backend/retrieval/` | Dense retrieval, Qdrant BM25 sparse retrieval, and RRF fusion. |
| `backend/reranker/` | BGE reranker loading and scoring. |
| `backend/correction/` | CRAG context grading and query rewriting. |
| `backend/generation/` | Text prompts, LLM providers, text answers, and visual answers. |
| `backend/visual/` | ColPali page rendering, indexing, visual search, and page-image storage. |
| `backend/evaluation/` | SQLite logging and Ragas-style scoring. |
| `backend/db/` | Qdrant collection setup, upserts, searches, and document listing. |

### Frontend

Streamlit uses a multipage layout:

| File | Purpose |
| --- | --- |
| `frontend/app.py` | App shell and page setup. |
| `frontend/pages/0_Chatbot.py` | Upload, searchable library, chat, answer details, and evidence display. |
| `frontend/pages/1_Evaluation.py` | Evaluation metrics, logged query table, and recompute button. |

The chatbot page should stay clean. Detailed metadata belongs inside the
collapsed `Answer details` panel.

## Ingestion Flow

`POST /ingest` does this:

1. Save and repair the uploaded PDF.
2. Analyze the PDF with `backend/ingestion/pdf_analyzer.py`.
3. Parse text, OCR, and tables with `backend/ingestion/parsers.py`.
4. Chunk parsed documents with `backend/ingestion/chunker.py`.
5. Create dense embeddings with `BAAI/bge-base-en-v1.5` by default.
6. Create sparse BM25 vectors with Qdrant FastEmbed.
7. Upsert text chunks into the Qdrant text collection.
8. Decide whether visual indexing is needed.
9. If needed, render pages, create ColPali embeddings, upsert visual pages, and
   store page images under `storage/visual_pages/`.

Visual indexing is controlled by:

```env
VISUAL_INDEX_MODE=auto
ENABLE_VISUAL_FALLBACK=true
```

Supported `VISUAL_INDEX_MODE` values:

- `auto`: index visually only when the PDF looks scanned, image-heavy, low-text,
  or layout-heavy.
- `always`: force visual indexing for every PDF.
- `never`: disable visual indexing.

## Query Flow

`POST /query` does this:

1. Handle simple greetings locally.
2. Handle library inventory questions from `/documents`.
3. Retrieve text evidence with BM25 sparse search and dense vector search.
4. Fuse candidates with reciprocal rank fusion.
5. Rerank with the configured BGE reranker.
6. Grade retrieved context with the CRAG grader.
7. Rewrite and re-retrieve if the grade is weak.
8. Generate a text answer if text evidence is good enough.
9. If text evidence is weak and visual pages exist, use ColPali visual retrieval
   and a multimodal LLM answer.
10. Log the query for evaluation and compute scores in the background.

The main answer path should return:

```json
{
  "answer": "...",
  "citations": [],
  "chunks": [],
  "visual_results": [],
  "answer_mode": "text_hybrid",
  "retrieval_summary": {},
  "was_corrected": false,
  "grade_score": 0.9,
  "original_query": "...",
  "query_used": "..."
}
```

## Prompts

Central prompt file:

```text
backend/generation/prompt.py
```

Keep these there:

- main RAG system prompt
- internal grading prompt
- query rewrite prompt
- visual QA prompt
- Ragas judge prompt

Do not put long prompt strings in `main.py`, `grader.py`, `rewriter.py`,
`answer_gen.py`, `vision_answer_gen.py`, or `ragas_eval.py`.

## Configuration

Common `.env` values:

```env
QDRANT_COLLECTION_NAME=Retriva
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=bm25

EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_VECTOR_SIZE=768
BM25_MODEL=qdrant/bm25
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_USE_FP16=false

VISUAL_INDEX_MODE=auto
ENABLE_VISUAL_FALLBACK=true
COLPALI_MODEL=vidore/colpali-v1.2

LLM_PROVIDER=openrouter
OPENROUTER_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
VISUAL_OPENROUTER_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

When changing runtime model names or feature flags, update `.env` if the user
asks for actual behavior changes. Update `.env.example` only for documented
defaults.

## Qdrant Notes

Text evidence is stored in the configured text collection with:

- dense vector name from `QDRANT_DENSE_VECTOR_NAME`
- sparse vector name from `QDRANT_SPARSE_VECTOR_NAME`
- full chunk payloads, including `source`, `page`, and `chunk_id`

Visual evidence is stored in the configured ColPali collection with:

- named vector `colpali`
- page-level payloads, including `source`, `page`, and image path

The sidebar library comes from Qdrant through `GET /documents`, not Streamlit
session state.

## Evaluation Notes

Evaluation is separate from the main pipeline.

`backend/evaluation/logger.py` stores query rows in SQLite.
`backend/evaluation/ragas_eval.py` computes:

- faithfulness
- answer relevancy
- context precision
- context recall

Evaluation runs in FastAPI background tasks so chat responses stay responsive.
If scores are pending, the frontend Evaluation page can queue recomputation.

## Latency Notes

Expected slow points:

- first model load after backend restart
- dense embedding during ingestion
- BGE reranking during query
- ColPali indexing for visual pages
- free OpenRouter model rate limits or slow provider responses

Known important detail:

```text
retrieval.rerank
```

was observed as the biggest query-time bottleneck with the larger reranker. The
current default is the smaller `BAAI/bge-reranker-base`. Keep the reranker in the
pipeline unless the user explicitly decides to remove that quality layer.

## UI Guidelines

- Keep the chat answer readable, like a normal assistant response.
- Keep source filenames out of the visible answer when possible; show compact
  citations in chat and full filenames in details.
- Put retrieval path, grade score, query rewrite, sources, chunks, and visual
  evidence inside `Answer details`.
- Keep the sidebar library searchable and collapsed when it may grow.
- Do not reintroduce manual retrieval-mode choices for normal user flow.

## Files Usually Not To Commit

- `.env`
- `.cache/`
- `.venv/`
- `qdrant_data/`
- `storage/visual_pages/`
- `backend/evaluation/eval_log.db`
- `__pycache__/`

Check `.gitignore` before adding generated files.

## Before Finishing A Change

For code changes, run at least a syntax check:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m compileall backend frontend
```

For UI changes, start the app if practical:

```powershell
streamlit run frontend/app.py
```

For backend behavior changes, check the relevant endpoint with FastAPI running:

```powershell
uvicorn backend.main:app
```

