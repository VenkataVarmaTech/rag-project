"""
BM25 Sparse Lexical Search Module.
Performs exact-keyword and token-frequency retrieval using BM25Okapi.
Complements dense vector retrieval for precise terms, acronyms, and product codes.
"""

import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25SearchIndex:
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple lowercase alphanumeric tokenization."""
        return re.findall(r"\b\w+\b", text.lower())

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Indexes a list of structured chunk dictionaries."""
        self.chunks = chunks
        self.corpus_tokens = [self.tokenize(c["text"]) for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Queries the BM25 index and returns top-k matching chunks."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:  # Only return documents with matching lexical terms
                c = self.chunks[idx]
                results.append({
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "score": score,
                    "metadata": c.get("metadata", {}),
                    "retrieval_source": "sparse"
                })
        return results

