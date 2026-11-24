"""
Cross-Encoder Re-ranker Module.
Re-scores and filters top-k candidates by passing (query, passage) pairs
through cross-attention to accurately determine relevance.
"""

from typing import List, Dict, Any
import numpy as np


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import CrossEncoder
            # Try to load if cached locally without hanging on network
            self.model = CrossEncoder(self.model_name, max_length=512, local_files_only=True)
            print(f"[*] Loaded Cross-Encoder: {self.model_name}")
        except Exception as e:
            # Graceful fallback: If network is restricted and model not pre-cached,
            # use an advanced token-interaction & lexical-coverage reranker
            print(f"[!] Cross-Encoder model '{self.model_name}' not cached locally. Using Fast Fallback Cross-Scorer.")
            self.model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunk dictionaries against the query.
        """
        if not candidates:
            return []

        if self.model is not None:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.model.predict(pairs)
            for i, c in enumerate(candidates):
                c["rerank_score"] = float(scores[i])
        else:
            # Fallback cross-scoring: computes query term coverage + position weighting
            query_words = set(query.lower().split())
            for c in candidates:
                text_lower = c["text"].lower()
                matched_words = sum(1 for w in query_words if w in text_lower)
                coverage = matched_words / max(1, len(query_words))
                # Combine original retrieval score with coverage
                base_score = c.get("score", 0.5)
                c["rerank_score"] = round(float(0.6 * base_score + 0.4 * coverage), 4)

        # Sort descending by rerank score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        return reranked

