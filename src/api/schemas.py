"""
Pydantic Schemas for FastAPI REST API Contracts.
Strict data validation, serialization, and Swagger API documentation schemas.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# INGESTION SCHEMAS
# ------------------------------------------------------------------------------
class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Path to TXT, MD, or PDF file to ingest")
    chunk_size: Optional[int] = Field(default=400, ge=100, le=2000, description="Max character length per chunk")
    chunk_overlap: Optional[int] = Field(default=80, ge=0, le=500, description="Character overlap between adjacent chunks")
    reset_collection: Optional[bool] = Field(default=False, description="Clear existing database before ingesting")


class IngestResponse(BaseModel):
    status: str
    file_name: str
    total_pages_or_sections: int
    total_chunks_created: int
    processing_time_ms: float


# ------------------------------------------------------------------------------
# SEARCH SCHEMAS
# ------------------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    mode: Literal["hybrid", "dense", "sparse"] = Field(default="hybrid", description="Retrieval strategy")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of candidate chunks to retrieve")


class SearchResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    retrieval_source: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_results: int
    latency_ms: float
    results: List[SearchResultItem]


# ------------------------------------------------------------------------------
# FULL RAG QUERY SCHEMAS
# ------------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question for the RAG assistant")
    mode: Literal["hybrid", "dense", "sparse"] = Field(default="hybrid", description="Retrieval strategy")
    use_reranker: bool = Field(default=True, description="Apply Cross-Encoder re-ranking to candidate chunks")
    top_k: int = Field(default=5, ge=1, le=50, description="Initial retrieval candidate count")
    rerank_top_k: int = Field(default=3, ge=1, le=10, description="Final chunks given to the generator")


class CitationItem(BaseModel):
    source_index: int
    document: str
    page: int
    chunk_id: str
    relevance_score: float
    text_snippet: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationItem]
    retrieval_mode: str
    reranked: bool
    latency_ms: float


# ------------------------------------------------------------------------------
# EVALUATION & STATS SCHEMAS
# ------------------------------------------------------------------------------
class EvaluationResponse(BaseModel):
    summary: Dict[str, Any]
    cases: List[Dict[str, Any]]
    latency_ms: float


class SystemStatsResponse(BaseModel):
    status: str
    chroma_vector_count: int
    bm25_chunk_count: int
    embedding_dimension: int
    embedding_model: str

