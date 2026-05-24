# Retriva Project Report

This document explains Retriva from start to finish: why it was built, how to
start it, how the system works, which stages were implemented, what problems
appeared during development, and how they were solved.

Use this file as the complete project reference. Keep `README.md` concise and
portfolio-friendly; keep detailed build history here.

## 1. Project Overview

Retriva is a self-correcting hybrid document RAG system for PDFs.

The system allows a user to upload PDFs, index them, ask natural language
questions, and receive citation-grounded answers. It supports normal text PDFs,
scanned PDFs, tables, forms, charts, image-heavy documents, and layout-heavy
pages.

The current system combines:

- text extraction
- OCR
- table extraction
- text chunking
- dense vector retrieval
- Qdrant BM25 sparse retrieval
- reciprocal rank fusion
- BGE reranking
- CRAG-style self-correction
- ColPali visual page retrieval
- multimodal answer generation
- Ragas-style evaluation
- Qdrant-backed document library
- Streamlit chatbot UI

## 2. Problem Statement

Basic PDF chat systems usually work only when the PDF contains clean embedded
text. Real documents are more difficult. They can contain:

- scanned pages
- signatures
- forms
- score cards
- screenshots
- charts
- tables
- images
- mixed text and layout

The main goal of Retriva is to make the system automatically choose the right
retrieval path, instead of asking the user to understand OCR, BM25, dense
embeddings, visual retrieval, or reranking.

The intended user flow is simple:

```text
Upload PDF -> Ingest document -> Ask a question -> Get a cited answer
```

## 3. Final Capabilities

Retriva currently supports:

| Capability | Status |
| --- | --- |
| Text PDF ingestion | Complete |
| Scanned PDF OCR | Complete |
| Table extraction | Complete |
| Chunking | Complete |
| Dense retrieval | Complete |
| Qdrant BM25 sparse retrieval | Complete |
| RRF fusion | Complete |
| BGE reranking | Complete |
| CRAG-style self-correction | Complete |
| Query rewriting | Complete |
| Answer generation with citations | Complete |
| Streamlit chatbot UI | Complete |
| Evaluation dashboard | Complete |
| Ragas-style scoring | Complete |
| ColPali visual retrieval | Complete |
| Multimodal visual answer generation | Complete |
| Automatic visual fallback | Complete |
| Searchable document library | Complete |
| Recruiter-friendly README | Complete |
| Agent guide | Complete |

## 4. Tech Stack

| Layer | Technology |
| --- | --- |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Vector database | Qdrant |
| PDF parsing | PyMuPDF |
| OCR | Tesseract, Pillow |
| Table extraction | Camelot |
| Dense embeddings | Sentence Transformers, BGE |
| Sparse retrieval | Qdrant FastEmbed BM25 |
| Fusion | Reciprocal rank fusion |
| Reranking | FlagEmbedding BGE reranker |
| Visual retrieval | ColPali |
| LLM provider | OpenRouter / OpenAI-compatible API |
| Evaluation | Ragas-style metrics, SQLite |
| Config | python-dotenv and `.env` |

## 5. Project Structure

```text
retriva/
|-- backend/
|   |-- main.py
|   |-- pdf_utils.py
|   |-- correction/
|   |   |-- grader.py
|   |   |-- pipeline.py
|   |   `-- rewriter.py
|   |-- db/
|   |   `-- qdrant_client.py
|   |-- evaluation/
|   |   |-- logger.py
|   |   `-- ragas_eval.py
|   |-- generation/
|   |   |-- answer_gen.py
|   |   |-- llm_provider.py
|   |   |-- prompt.py
|   |   `-- vision_answer_gen.py
|   |-- ingestion/
|   |   |-- chunker.py
|   |   |-- parsers.py
|   |   `-- pdf_analyzer.py
|   |-- reranker/
|   |   `-- bge_reranker.py
|   |-- retrieval/
|   |   |-- bm25_retriever.py
|   |   |-- dense_retriever.py
|   |   `-- fusion.py
|   `-- visual/
|       |-- colpali_retriever.py
|       `-- page_renderer.py
|-- frontend/
|   |-- app.py
|   `-- pages/
|       |-- 0_Chatbot.py
|       `-- 1_Evaluation.py
|-- storage/
|   `-- visual_pages/
|-- AGENTS.md
|-- PROJECT_REPORT.md
|-- README.md
|-- docker-compose.yml
|-- requirements.txt
`-- .env.example
```

## 6. How To Start The Project

### Step 1: Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```powershell
uv pip install -r requirements.txt
```

Regular pip can also be used:

```powershell
pip install -r requirements.txt
```

### Step 3: Create `.env`

```powershell
Copy-Item .env.example .env
```

Set at least:

```env
QDRANT_URL=https://your-qdrant-cloud-url
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION_NAME=Retriva

LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
VISUAL_OPENROUTER_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

### Step 4: Start the backend

```powershell
uvicorn backend.main:app
```

When editing backend files:

```powershell
uvicorn backend.main:app --reload --reload-dir backend
```

Avoid plain `--reload` across the whole repo because it reloads the backend
when frontend files change and forces large models to load again.

### Step 5: Start the frontend

```powershell
streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

### Step 6: Use the app

1. Upload a PDF.
2. Click `Ingest document`.
3. Search the library to confirm the document is indexed.
4. Ask a question.
5. Read the answer.
6. Open `Answer details` when evidence or retrieval metadata is needed.

## 7. API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/ingest` | `POST` | Upload and index a PDF. |
| `/query` | `POST` | Ask a question and get a cited answer. |
| `/documents` | `GET` | List indexed documents from Qdrant. |
| `/eval_logs` | `GET` | Read evaluation logs. |
| `/eval_logs/recompute` | `POST` | Recompute pending evaluation scores. |
| `/ingest_visual` | `POST` | Debug endpoint for visual-only indexing. |
| `/query_visual` | `POST` | Debug endpoint for text-vs-visual comparison. |

## 8. Build Stages

### Stage 1: Base Hybrid RAG Pipeline

Goal:

Build a working pipeline that can ingest PDFs, chunk them, retrieve evidence,
rerank it, and generate cited answers through FastAPI.

Implemented:

- `POST /ingest`
- `POST /query`
- PyMuPDF text extraction
- Tesseract OCR
- Camelot table extraction
- simple overlapping character chunking
- BGE dense embeddings
- Qdrant storage
- BM25 sparse retrieval
- dense retrieval
- RRF fusion
- BGE reranking
- OpenAI/OpenRouter-compatible LLM provider
- answer generation with citations
- Streamlit upload and chat UI

Initial system flow:

```text
PDF -> parse -> chunk -> embed -> Qdrant -> retrieve -> fuse -> rerank -> answer
```

### Stage 2: CRAG-Style Self-Correction

Goal:

Add a correction layer between retrieval and answer generation.

Implemented:

- context grading
- query rewriting
- re-retrieval when context quality is weak
- response fields:
  - `query_used`
  - `was_corrected`
  - `grade_score`

Correction flow:

```text
retrieve -> rerank -> grade context -> rewrite if weak -> retrieve again
```

This made the system more robust when the first retrieved chunks were not strong
enough.

### Stage 3: Evaluation Dashboard

Goal:

Log every query and compute evaluation scores.

Implemented:

- SQLite evaluation database
- query logging
- Ragas-style metric computation
- background evaluation tasks
- Streamlit evaluation page
- recompute pending scores button

Metrics:

- faithfulness
- answer relevancy
- context precision
- context recall

Current measured evaluation result:

| Metric | Score |
| --- | ---: |
| Faithfulness | 98.57% |
| Answer relevancy | 94.79% |
| Context precision | 91.71% |
| Context recall | 100.00% |
| Average RAG quality | 96.27% |

Important note:

This score is based on the current logged document-QA evaluation set. It is not
a universal benchmark. It should be updated as the evaluation set grows.

### Stage 4: Visual Retrieval With ColPali

Goal:

Add visual page retrieval for documents where text extraction alone is not
enough.

Implemented:

- `backend/visual/`
- PDF page rendering
- ColPali page embeddings
- visual Qdrant collection
- visual page search
- saved page images
- visual comparison endpoints
- multimodal visual answer generation

Initial visual mode only returned relevant pages. Later, this was improved so
the system could generate answers from retrieved page images using a multimodal
LLM.

### Stage 5: Automatic Routing and Product Polish

Goal:

Make the system feel like a simple product instead of a technical demo with
manual modes.

Implemented:

- automatic visual indexing based on PDF analysis
- automatic visual fallback during query
- local greeting handling
- document inventory intent handling
- `/documents` endpoint
- searchable Qdrant-backed library
- collapsed `Answer details`
- compact citations in chat
- cleaner chatbot UI
- separate evaluation page
- centralized prompts
- improved README
- `AGENTS.md` project guide
- config alignment across `.env.example`, backend defaults, and Docker Compose

Final user flow:

```text
Upload any PDF -> Retriva indexes the right evidence -> Ask naturally -> Get answer
```

## 9. Final System Flow

### Ingestion Flow

```text
Upload PDF
  -> save file
  -> repair/normalize PDF
  -> analyze PDF
  -> parse text
  -> OCR scanned pages if needed
  -> extract tables
  -> chunk documents
  -> create dense BGE embeddings
  -> create Qdrant BM25 sparse vectors
  -> upsert text chunks to Qdrant
  -> decide if visual indexing is needed
  -> render visual pages
  -> create ColPali page embeddings
  -> upsert visual pages to Qdrant
```

### Query Flow

```text
User question
  -> handle greeting or library question if applicable
  -> dense retrieval
  -> BM25 sparse retrieval
  -> RRF fusion
  -> BGE reranking
  -> CRAG context grading
  -> query rewrite if weak
  -> re-retrieve if needed
  -> visual fallback if text evidence is weak
  -> generate answer
  -> extract citations
  -> log evaluation row
  -> compute evaluation metrics in background
```

## 10. How Qdrant Is Used

Retriva uses Qdrant as the central evidence store.

Text chunks are stored with:

- dense vector named `dense`
- sparse vector named `bm25`
- payload with text, page, source, chunk id, and metadata

Visual pages are stored separately with:

- ColPali page embeddings
- page number
- source filename
- rendered image path

The Streamlit library reads from Qdrant through `/documents`, so it reflects
what is actually indexed.

## 11. Prompt Organization

All main prompts are centralized in:

```text
backend/generation/prompt.py
```

This file contains:

- main answer system prompt
- internal grading prompt
- query rewrite prompt
- visual answer prompt
- Ragas judge prompt

This avoids duplicated prompt text across the codebase.

## 12. Problems Faced And Solutions

### Problem 1: Dependency warning from Camelot extra

Issue:

`camelot-py[cv]` produced a warning because the installed version did not expose
the `cv` extra in the expected way.

Solution:

Use `camelot-py` directly in `requirements.txt`. OpenCV dependencies were still
resolved by the dependency graph where needed.

### Problem 2: Environment variable handling became too flexible

Issue:

An early helper accepted too many alternate environment variable names. This
made the config harder to understand.

Solution:

Simplified the configuration and used `python-dotenv` consistently. Current
runtime configuration comes from `.env`, with documented defaults in
`.env.example`.

### Problem 3: BM25 was initially in memory

Issue:

An in-memory BM25 index is simple, but it does not match the goal of using
Qdrant as the retrieval store.

Solution:

Moved BM25 into Qdrant using sparse vectors and FastEmbed. The final retrieval
uses Qdrant for both dense and sparse search.

### Problem 4: Reranker compatibility error

Issue:

The first reranker setup produced a tokenizer compatibility error:

```text
XLMRobertaTokenizer has no attribute prepare_for_model
```

Solution:

Aligned transformer-related dependencies and reranker usage so the BGE reranker
worked correctly. Later, the reranker model was changed to a smaller default for
better local latency:

```env
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_USE_FP16=false
```

### Problem 5: First model load was slow

Issue:

Embedding, reranking, and ColPali models are large. Even after they are
downloaded, they still need time to load into memory after backend restart.

Solution:

Added model caching under:

```text
.cache/models
```

Also avoided full-repo backend reloads during frontend work:

```powershell
uvicorn backend.main:app --reload --reload-dir backend
```

### Problem 6: Greeting questions were answered like document questions

Issue:

When the user sent `hello`, the strict document-grounded prompt refused to
answer because the context did not contain greeting information.

Solution:

Added local greeting detection in the backend so simple greetings receive a
friendly response without using the RAG pipeline.

### Problem 7: Citation format looked cluttered in chat

Issue:

Full citation tags such as:

```text
[Source: page 1, filename.pdf]
```

made the answer visually heavy.

Solution:

Kept full source metadata in the backend response and `Answer details`, while
displaying compact chat citations like:

```text
[p. 1]
```

### Problem 8: Ragas scores stayed pending

Issue:

Ragas evaluation ran in the background and some metrics stayed pending because
the default evaluator setup did not always return all metric values.

Solution:

Added OpenRouter-based judge scoring as a fallback and a recompute button on the
Evaluation page. The system now fills faithfulness, answer relevancy, context
precision, and context recall more reliably.

### Problem 9: ColPali indexing timed out while writing to Qdrant

Issue:

Visual page embeddings are large. Uploading too much at once caused Qdrant write
timeouts.

Solution:

Added visual upsert batching and configurable Qdrant visual timeout:

```env
COLPALI_UPSERT_BATCH_SIZE=1
QDRANT_VISUAL_TIMEOUT=300
```

### Problem 10: Visual retrieval returned pages, not answers

Issue:

The first visual path returned relevant pages and scores, but did not generate a
final answer.

Solution:

Added multimodal answer generation. The system now retrieves visual pages,
sends page images to a vision-capable LLM, and returns a cited answer.

### Problem 11: Manual retrieval modes made the UX confusing

Issue:

The UI originally exposed text, visual, and compare modes. This was useful for
debugging but not ideal for normal users.

Solution:

Changed the system to automatic routing:

- always ingest text
- visually index only when useful
- use visual fallback only when text evidence is weak

The debug endpoints remain available for development.

### Problem 12: User asked which documents were stored, but RAG answered from chunks

Issue:

The question `which docs do you have?` should be answered from the database
inventory, not retrieved document text.

Solution:

Added `/documents`, Qdrant-backed document listing, and library-intent handling.
The app now answers inventory questions directly from indexed document metadata.

### Problem 13: Config defaults drifted across files

Issue:

Some old defaults remained in code and Docker Compose, including old model names
and an old collection name.

Solution:

Aligned defaults across:

- backend code
- `.env.example`
- `docker-compose.yml`
- `README.md`
- `AGENTS.md`

Final default examples:

```env
QDRANT_COLLECTION_NAME=Retriva
RERANKER_MODEL=BAAI/bge-reranker-base
OPENROUTER_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
VISUAL_OPENROUTER_MODEL=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
```

## 13. Current Evaluation Result

The current local evaluation logs show the following document-QA quality:

| Metric | Score |
| --- | ---: |
| Faithfulness | 98.57% |
| Answer relevancy | 94.79% |
| Context precision | 91.71% |
| Context recall | 100.00% |
| Average RAG quality | 96.27% |

Important:

This score is based on the current logged evaluation set. It is an internal
measurement, not a universal benchmark. A larger benchmark set should be built
for stronger reporting.

## 14. Current Configuration Defaults

Important `.env` values:

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

ENABLE_CRAG=true
ENABLE_QUERY_REWRITE=true
CRAG_GRADE_THRESHOLD=0.7
```

## 15. Validation Performed

During the final codebase check:

- Python syntax parse passed for `backend/` and `frontend/`.
- Stale config names were removed.
- `.env.example` was aligned with backend environment variables.
- Docker Compose defaults were aligned with the current system.
- README was rewritten for recruiter visibility.
- AGENTS.md was added for future coding agents.
- Documentation files were kept ASCII-safe.

## 16. What To Improve Next

Good future improvements:

1. Build a larger evaluation benchmark with 50 to 100 question-answer pairs.
2. Add document-level filtering so users can ask questions against selected PDFs.
3. Add streaming responses in the Streamlit UI.
4. Add authentication for multi-user deployments.
5. Add async job tracking for long visual ingestion.
6. Add page image previews in answer details.
7. Add exportable evaluation reports.
8. Add Docker GPU profiles for faster local inference.
9. Add CI checks for syntax, formatting, and config drift.
10. Add automated smoke tests for `/ingest`, `/query`, and `/documents`.

## 17. Final Summary

Retriva started as a Stage 1 PDF RAG pipeline and evolved into a more complete
document intelligence system.

The final system can:

- ingest mixed PDF types
- retrieve with dense and sparse search
- rerank evidence
- self-correct weak retrieval
- use visual retrieval when needed
- generate citation-grounded answers
- evaluate answer quality
- show indexed documents in a searchable library
- provide a clean Streamlit chatbot experience

The key design principle is:

```text
Keep the interface simple, but make the retrieval system capable underneath.
```
