# Enterprise Hybrid RAG System with Cross-Encoder Re-Ranking & FastAPI

A production-grade, asynchronous **Retrieval-Augmented Generation (RAG)** backend microservice built with **FastAPI**, **ChromaDB**, **BM25**, and **Sentence-Transformers**.

---

## 🌟 Key Architecture & Highlights

* **Hybrid Retrieval (Dense + Sparse):** Merges dense semantic vector search (**ChromaDB** with `all-MiniLM-L6-v2`) and sparse keyword search (**BM25Okapi**) via **Reciprocal Rank Fusion (RRF)** to eliminate vocabulary mismatch.
* **Two-Stage Retrieval Pipeline:** Integrates a **Cross-Encoder re-ranking stage** to filter false positives and context noise before prompt synthesis.
* **Recursive Character Chunking:** Preserves semantic paragraph and sentence boundaries with sliding window overlaps (`\n\n` → `\n` → `. ` → ` `).
* **Automated Evaluation Suite (The RAG Triad):** Built-in benchmarking engine measuring **Hit Rate @ K**, **Mean Reciprocal Rank (MRR)**, **Context Relevance**, and **Faithfulness (Groundedness)** against golden datasets.
* **Production-Grade REST API:** Asynchronous **FastAPI** backend with strict Pydantic V2 request/response contracts, process-time middleware, and CORS configuration.

---

## 🏗️ System Architecture

```
                               ┌────────────────────────────────┐
                               │       Client / UI / REST       │
                               └───────────────┬────────────────┘
                                               │ HTTP / JSON
                                               ▼
                               ┌────────────────────────────────┐
                               │        FastAPI Backend         │
                               │  /ingest  /search  /query      │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
     [Dense Semantic Store]                                          [Sparse Keyword Index]
    ChromaDB (HNSW Graph, 384D)                                     BM25 (Exact Token Frequency)
               │                                                               │
               └───────────────────────────────┬───────────────────────────────┘
                                               ▼
                              [Reciprocal Rank Fusion (RRF)]
                                               │
                                               ▼
                             [Cross-Encoder Re-ranking Stage]
                                               │
                                               ▼
                           [Grounded Synthesis & Source Citations]
```

---

## 📁 Repository Structure

```
.
├── 01_bare_metal_rag/         # Foundational baseline using pure Python & NumPy
│   └── bare_rag.py
├── data/                      # Knowledge base documents and evaluation benchmarks
│   ├── knowledge_base.txt     # Sample operational technical manual
│   └── eval_ground_truth.json # Golden Q&A test pairs
├── src/                       # Application source code
│   ├── config.py              # Central hyperparameters and environment settings
│   ├── core/                  # Core RAG algorithmic components
│   │   ├── document_loader.py # Ingestion for TXT, Markdown, and PDF
│   │   ├── chunker.py         # Recursive character text splitter
│   │   ├── embeddings.py      # Dense vector embedding manager (Singleton)
│   │   ├── vector_store.py    # Persistent ChromaDB wrapper
│   │   ├── bm25_index.py      # BM25 sparse keyword index
│   │   ├── hybrid_retriever.py# Dense + Sparse fusion with RRF
│   │   ├── reranker.py        # Cross-Encoder re-ranker
│   │   └── generator.py       # Grounded prompt synthesis & source citations
│   ├── evaluation/            # Automated quality assurance & metrics
│   │   └── evaluator.py       # Hit Rate, MRR, Context Relevance & Faithfulness
│   └── api/                   # REST API layer (FastAPI)
│       ├── schemas.py         # Strict Pydantic V2 validation models
│       ├── routes.py          # API route handlers
│       └── server.py          # Server entrypoint with lifespan & middleware
└── tests/                     # Integration tests and automated benchmark runners
    └── test_pipeline.py
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
* Python 3.10+
* Git

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/VenkataVarmaTech/rag-project.git
cd rag-project
python -m pip install -r requirements.txt
```

### 3. Run the FastAPI Server
```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation will be available at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 📡 API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Service health check and indexed vector count |
| `POST` | `/api/v1/documents/ingest` | Ingest and index TXT, MD, or PDF documents |
| `POST` | `/api/v1/search` | Compare dense, sparse, and hybrid retrieval side-by-side |
| `POST` | `/api/v1/query` | Full RAG query pipeline with re-ranking and citations |
| `POST` | `/api/v1/evaluate` | Run automated RAG evaluation benchmark |
| `GET` | `/api/v1/stats` | Retrieve index statistics and embedding model metadata |

---

## 📊 Evaluation Metrics

The system includes automated benchmarking via `RAGEvaluator`:
* **Hit Rate @ K:** Measures whether relevant ground-truth chunks appear in the top-$K$ candidates.
* **Mean Reciprocal Rank (MRR):** Measures the ranking position of the first relevant chunk.
* **Context Relevance:** Evaluates the signal-to-noise ratio in retrieved context blocks.
* **Faithfulness / Groundedness:** Detects hallucinations by verifying that generated answers are directly supported by retrieved context.

---

## 👤 Author
* **Vatsavayi Venkata Varma**
* LinkedIn / GitHub: [github.com/VenkataVarmaTech](https://github.com/VenkataVarmaTech)
