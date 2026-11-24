"""
FastAPI Route Handlers.
Exposes endpoints for Ingestion, Search, Full RAG Queries, System Stats, and Evaluation.
"""

import time
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from src.config import DATA_DIR, EMBEDDING_MODEL_NAME
from src.core.document_loader import DocumentLoader
from src.core.chunker import RecursiveCharacterChunker
from src.core.embeddings import DenseEmbedder
from src.core.vector_store import ChromaVectorStore
from src.core.bm25_index import BM25SearchIndex
from src.core.hybrid_retriever import HybridRetriever
from src.core.reranker import CrossEncoderReranker
from src.core.generator import GroundedGenerator
from src.evaluation.evaluator import RAGEvaluator
from src.api.schemas import (
    IngestRequest, IngestResponse,
    SearchRequest, SearchResponse, SearchResultItem,
    QueryRequest, QueryResponse, CitationItem,
    EvaluationResponse, SystemStatsResponse
)


router = APIRouter(prefix="/api/v1")


class RAGService:
    """Singleton-style service container managing the RAG lifecycle."""
    def __init__(self):
        print("[*] Initializing RAG Service...")
        self.embedder = DenseEmbedder()
        self.vector_store = ChromaVectorStore()
        self.bm25_index = BM25SearchIndex()
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            embedder=self.embedder
        )
        self.reranker = CrossEncoderReranker()
        self.generator = GroundedGenerator()
        self.all_chunks = []

    def ingest_document(self, file_path_str: str, chunk_size: int = 400, chunk_overlap: int = 80, reset: bool = False):
        start_time = time.perf_counter()
        file_path = Path(file_path_str)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path_str}")

        if reset:
            self.vector_store.reset()
            self.all_chunks = []

        # 1. Load
        raw_docs = DocumentLoader.load_file(file_path)

        # 2. Chunk
        chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_documents(raw_docs)

        # 3. Embed
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # 4. Index in Vector Store
        self.vector_store.add_chunks(chunks, embeddings.tolist())

        # 5. Index in BM25
        self.all_chunks.extend(chunks)
        self.bm25_index.build_index(self.all_chunks)

        duration_ms = (time.perf_counter() - start_time) * 1000
        return {
            "status": "success",
            "file_name": file_path.name,
            "total_pages_or_sections": len(raw_docs),
            "total_chunks_created": len(chunks),
            "processing_time_ms": round(duration_ms, 2)
        }


# Global instance
service = RAGService()


@router.post("/documents/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(request: IngestRequest):
    try:
        result = service.ingest_document(
            file_path_str=request.file_path,
            chunk_size=request.chunk_size or 400,
            chunk_overlap=request.chunk_overlap or 80,
            reset=request.reset_collection or False
        )
        return IngestResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_chunks(request: SearchRequest):
    start_time = time.perf_counter()
    try:
        results = service.hybrid_retriever.retrieve(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k
        )
        duration_ms = (time.perf_counter() - start_time) * 1000

        formatted_items = [
            SearchResultItem(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=round(r.get("score", 0.0), 4),
                retrieval_source=r.get("retrieval_source", request.mode),
                metadata=r.get("metadata", {})
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            mode=request.mode,
            total_results=len(formatted_items),
            latency_ms=round(duration_ms, 2),
            results=formatted_items
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def answer_query(request: QueryRequest):
    start_time = time.perf_counter()
    try:
        # Step 1: Candidate Retrieval (Hybrid/Dense/Sparse)
        candidates = service.hybrid_retriever.retrieve(
            query=request.query,
            mode=request.mode,
            top_k=request.top_k
        )

        # Step 2: Cross-Encoder Re-ranking (if requested)
        if request.use_reranker and candidates:
            final_chunks = service.reranker.rerank(
                query=request.query,
                candidates=candidates,
                top_k=request.rerank_top_k
            )
        else:
            final_chunks = candidates[:request.rerank_top_k]

        # Step 3: Grounded Generation
        generation_res = service.generator.generate_answer(
            query=request.query,
            chunks=final_chunks
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        citations = [
            CitationItem(
                source_index=c["source_index"],
                document=c["document"],
                page=c["page"],
                chunk_id=c["chunk_id"],
                relevance_score=c["relevance_score"],
                text_snippet=c["text_snippet"]
            )
            for c in generation_res["citations"]
        ]

        return QueryResponse(
            query=request.query,
            answer=generation_res["answer"],
            citations=citations,
            retrieval_mode=request.mode,
            reranked=request.use_reranker,
            latency_ms=round(duration_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation():
    start_time = time.perf_counter()
    ground_truth_file = DATA_DIR / "eval_ground_truth.json"
    if not ground_truth_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark ground truth dataset not found.")

    evaluator = RAGEvaluator(ground_truth_file)

    def retriever_fn(q, top_k=3):
        candidates = service.hybrid_retriever.retrieve(q, mode="hybrid", top_k=top_k * 2)
        return service.reranker.rerank(q, candidates, top_k=top_k)

    def generator_fn(q, chunks):
        return service.generator.generate_answer(q, chunks)

    benchmark_res = evaluator.run_benchmark(retriever_fn, generator_fn, top_k=3)
    duration_ms = (time.perf_counter() - start_time) * 1000

    return EvaluationResponse(
        summary=benchmark_res["summary"],
        cases=benchmark_res["cases"],
        latency_ms=round(duration_ms, 2)
    )


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    return SystemStatsResponse(
        status="healthy",
        chroma_vector_count=service.vector_store.count(),
        bm25_chunk_count=len(service.all_chunks),
        embedding_dimension=service.embedder.dimension,
        embedding_model=EMBEDDING_MODEL_NAME
    )

