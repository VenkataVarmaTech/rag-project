"""
FastAPI Server Entrypoint.
Configures Middleware, Lifespan Hooks, OpenAPI Documentation, and Routes.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.config import DATA_DIR
from src.api.routes import router, service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Auto-index the sample knowledge base if the store is empty
    kb_path = DATA_DIR / "knowledge_base.txt"
    if service.vector_store.count() == 0 and kb_path.exists():
        print(f"[*] Lifespan: Auto-ingesting baseline knowledge base from {kb_path}...")
        service.ingest_document(str(kb_path), chunk_size=400, chunk_overlap=80)
    yield
    # Shutdown logic (if any)
    print("[*] Lifespan: Shutting down RAG API Server.")


app = FastAPI(
    title="Enterprise Hybrid RAG Backend Service",
    description="Production-Grade REST API featuring Hybrid Retrieval (BM25 + ChromaDB), Reciprocal Rank Fusion, Cross-Encoder Re-ranking, and Automated Evaluation.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend clients (e.g., Streamlit, React, Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


# Include API routes
app.include_router(router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Enterprise Hybrid RAG Backend API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health": "OK",
        "chroma_chunks_indexed": service.vector_store.count()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)

