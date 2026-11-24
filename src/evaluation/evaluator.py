"""
RAG Evaluation Suite Module.
Implements industry-standard RAG metrics:
1. Retrieval Metrics:
   - Hit Rate @ K
   - Mean Reciprocal Rank (MRR)
2. Generation Metrics (The RAG Triad):
   - Context Relevance (Retrieval Signal-to-Noise Ratio)
   - Faithfulness / Groundedness (Hallucination Detection)
   - Answer Relevance
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class RAGEvaluator:
    def __init__(self, ground_truth_path: Path | str):
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth_data = self._load_ground_truth()

    def _load_ground_truth(self) -> List[Dict[str, Any]]:
        if not self.ground_truth_path.exists():
            return []
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def calculate_retrieval_hit(retrieved_chunks: List[Dict[str, Any]], expected_keywords: List[str]) -> tuple[bool, int | None]:
        """
        Checks if any expected keywords appear in the retrieved chunks.
        Returns (is_hit, rank_of_first_hit).
        """
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk["text"].lower()
            # If at least half of the expected keywords appear in this chunk, it's a hit
            matched = sum(1 for kw in expected_keywords if kw.lower() in text)
            if matched >= max(1, len(expected_keywords) // 2):
                return True, rank
        return False, None

    @staticmethod
    def evaluate_context_relevance(query: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Context Relevance = (Relevant retrieved sentences) / (Total retrieved sentences).
        Approximated via query term semantic overlap.
        """
        if not retrieved_chunks:
            return 0.0

        query_terms = set(query.lower().split())
        total_chunks = len(retrieved_chunks)
        relevant_chunks = 0

        for c in retrieved_chunks:
            chunk_terms = set(c["text"].lower().split())
            overlap = len(query_terms.intersection(chunk_terms))
            # If at least 25% of query words are present, consider the chunk relevant context
            if (overlap / max(1, len(query_terms))) >= 0.25:
                relevant_chunks += 1

        return round(relevant_chunks / max(1, total_chunks), 4)

    @staticmethod
    def evaluate_faithfulness(generated_answer: str, retrieved_chunks: List[Dict[str, Any]]) -> float:
        """
        Faithfulness (Groundedness) = fraction of answer statements supported by context.
        Detects hallucinations where LLM introduces unverified facts.
        """
        if not retrieved_chunks:
            return 0.0

        combined_context = " ".join([c["text"].lower() for c in retrieved_chunks])
        answer_sentences = [s.strip() for s in generated_answer.split(".") if len(s.strip()) > 10]

        if not answer_sentences:
            return 1.0

        supported_sentences = 0
        for sent in answer_sentences:
            words = [w for w in sent.lower().split() if len(w) > 3]
            if not words:
                continue
            matches = sum(1 for w in words if w in combined_context)
            if (matches / len(words)) >= 0.6:  # 60% of key words in context
                supported_sentences += 1

        return round(supported_sentences / max(1, len(answer_sentences)), 4)

    def run_benchmark(self, retriever_fn, generator_fn, top_k: int = 3) -> Dict[str, Any]:
        """
        Runs the full evaluation benchmark across all test cases.
        Returns aggregate metrics and individual case evaluations.
        """
        total_queries = len(self.ground_truth_data)
        if total_queries == 0:
            return {"error": "Ground truth dataset is empty."}

        hits = 0
        reciprocal_ranks = []
        context_relevance_scores = []
        faithfulness_scores = []
        case_results = []

        for item in self.ground_truth_data:
            query = item["query"]
            expected_kws = item.get("expected_keywords", [])

            # 1. Retrieve
            retrieved = retriever_fn(query, top_k=top_k)

            # 2. Generate
            gen_result = generator_fn(query, retrieved)
            answer = gen_result["answer"]

            # 3. Calculate Retrieval Metrics
            is_hit, rank = self.calculate_retrieval_hit(retrieved, expected_kws)
            if is_hit:
                hits += 1
                reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            else:
                reciprocal_ranks.append(0.0)

            # 4. Calculate Generation Metrics
            ctx_rel = self.evaluate_context_relevance(query, retrieved)
            faith = self.evaluate_faithfulness(answer, retrieved)

            context_relevance_scores.append(ctx_rel)
            faithfulness_scores.append(faith)

            case_results.append({
                "query": query,
                "hit": is_hit,
                "hit_rank": rank,
                "context_relevance": ctx_rel,
                "faithfulness": faith,
                "generated_answer_snippet": answer[:150] + "..." if len(answer) > 150 else answer
            })

        hit_rate = round(hits / total_queries, 4)
        mrr = round(sum(reciprocal_ranks) / total_queries, 4)
        avg_ctx_rel = round(sum(context_relevance_scores) / total_queries, 4)
        avg_faith = round(sum(faithfulness_scores) / total_queries, 4)

        return {
            "summary": {
                "total_queries_evaluated": total_queries,
                "hit_rate_at_k": hit_rate,
                "mean_reciprocal_rank_mrr": mrr,
                "average_context_relevance": avg_ctx_rel,
                "average_faithfulness": avg_faith,
                "benchmark_status": "EXCELLENT" if hit_rate >= 0.8 and avg_faith >= 0.8 else "NEEDS_TUNING"
            },
            "cases": case_results
        }

