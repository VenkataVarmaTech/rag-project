"""
Hybrid Retriever Module.
Fuses Dense Semantic Search (ChromaDB) and Sparse Lexical Search (BM25)
using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Literal
from src.core.embeddings import DenseEmbedder
from src.core.vector_store import ChromaVectorStore
from src.core.bm25_index import BM25SearchIndex
from src.config import RRF_K_PARAMETER, DEFAULT_RETRIEVAL_TOP_K


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_index: BM25SearchIndex,
        embedder: DenseEmbedder,
        rrf_k: int = RRF_K_PARAMETER
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search_dense(self, query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_query(query).tolist()
        return self.vector_store.search(query_embedding, top_k=top_k)

    def search_sparse(self, query: str, top_k: int = DEFAULT_RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        return self.bm25_index.search(query, top_k=top_k)

    def search_hybrid(
        self,
        query: str,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        candidate_pool_size: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Executes both dense and sparse retrieval, then merges candidates via
        Reciprocal Rank Fusion (RRF).
        
        Formula:
            RRF_Score(doc) = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
        """
        # Retrieve larger candidate pools from each retriever
        dense_results = self.search_dense(query, top_k=candidate_pool_size)
        sparse_results = self.search_sparse(query, top_k=candidate_pool_size)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        # 1. Process Dense Ranks
        for rank, res in enumerate(dense_results, start=1):
            c_id = res["chunk_id"]
            dense_ranks[c_id] = rank
            chunk_map[c_id] = res
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 2. Process Sparse Ranks
        for rank, res in enumerate(sparse_results, start=1):
            c_id = res["chunk_id"]
            sparse_ranks[c_id] = rank
            if c_id not in chunk_map:
                chunk_map[c_id] = res
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # 3. Sort by aggregated RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        fused_results = []
        for c_id in sorted_chunk_ids:
            item = chunk_map[c_id].copy()
            item["score"] = round(rrf_scores[c_id], 6)
            item["retrieval_source"] = "hybrid"
            item["debug_info"] = {
                "dense_rank": dense_ranks.get(c_id, None),
                "sparse_rank": sparse_ranks.get(c_id, None),
                "rrf_score": rrf_scores[c_id]
            }
            fused_results.append(item)

        return fused_results

    def retrieve(
        self,
        query: str,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
        top_k: int = DEFAULT_RETRIEVAL_TOP_K
    ) -> List[Dict[str, Any]]:
        if mode == "dense":
            return self.search_dense(query, top_k=top_k)
        elif mode == "sparse":
            return self.search_sparse(query, top_k=top_k)
        else:
            return self.search_hybrid(query, top_k=top_k)

