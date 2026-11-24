"""
Comprehensive Integration & API Benchmark Test Suite.
Verifies all RAG components, FastAPI endpoints, and calculates evaluation metrics.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api.server import app
from src.config import DATA_DIR


def run_all_tests():
    print("=" * 80)
    print("  ENTERPRISE RAG SYSTEM: COMPREHENSIVE VERIFICATION & BENCHMARK SUITE")
    print("=" * 80)

    client = TestClient(app)

    # --------------------------------------------------------------------------
    # 1. Health & Root Endpoint Check
    # --------------------------------------------------------------------------
    print("\n[TEST 1] Verifying FastAPI Server Health Check...")
    root_resp = client.get("/")
    assert root_resp.status_code == 200, f"Root failed: {root_resp.text}"
    print(f"-> Status 200 OK | Response: {root_resp.json()}")

    # --------------------------------------------------------------------------
    # 2. Document Ingestion Test
    # --------------------------------------------------------------------------
    print("\n[TEST 2] Testing Document Ingestion Endpoint (/api/v1/documents/ingest)...")
    kb_path = DATA_DIR / "knowledge_base.txt"
    ingest_payload = {
        "file_path": str(kb_path),
        "chunk_size": 400,
        "chunk_overlap": 80,
        "reset_collection": True
    }
    ingest_resp = client.post("/api/v1/documents/ingest", json=ingest_payload)
    assert ingest_resp.status_code == 201, f"Ingest failed: {ingest_resp.text}"
    ingest_data = ingest_resp.json()
    print(f"-> Ingested: {ingest_data['file_name']}")
    print(f"-> Chunks Created: {ingest_data['total_chunks_created']} in {ingest_data['processing_time_ms']} ms")

    # --------------------------------------------------------------------------
    # 3. System Stats Verification
    # --------------------------------------------------------------------------
    print("\n[TEST 3] Verifying Index Stats (/api/v1/stats)...")
    stats_resp = client.get("/api/v1/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    print(f"-> ChromaDB Vectors: {stats['chroma_vector_count']} | BM25 Chunks: {stats['bm25_chunk_count']}")

    # --------------------------------------------------------------------------
    # 4. Dense vs Sparse vs Hybrid Retrieval Comparison
    # --------------------------------------------------------------------------
    print("\n[TEST 4] Comparing Dense vs Sparse vs Hybrid Retrieval (/api/v1/search)...")
    test_query = "What are the emergency isolation valves V-101 and V-102?"

    for mode in ["dense", "sparse", "hybrid"]:
        search_resp = client.post("/api/v1/search", json={"query": test_query, "mode": mode, "top_k": 2})
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        print(f"\n  * Mode: [{mode.upper()}] (Latency: {search_data['latency_ms']} ms)")
        for i, item in enumerate(search_data["results"], start=1):
            excerpt = item['text'].replace('\n', ' ')[:90]
            print(f"    {i}. [Score: {item['score']}] (Source: {item['retrieval_source']}) -> \"{excerpt}...\"")

    # --------------------------------------------------------------------------
    # 5. Full RAG Query with Citations & Reranking
    # --------------------------------------------------------------------------
    print("\n[TEST 5] Testing Full RAG Query with Cross-Encoder Reranking (/api/v1/query)...")
    query_payload = {
        "query": "What is the maximum allowed temperature for the mixing chamber before an alarm triggers?",
        "mode": "hybrid",
        "use_reranker": True,
        "top_k": 5,
        "rerank_top_k": 2
    }
    query_resp = client.post("/api/v1/query", json=query_payload)
    assert query_resp.status_code == 200
    q_data = query_resp.json()
    print(f"\nUser Query: \"{q_data['query']}\"")
    print(f"Latency: {q_data['latency_ms']} ms | Reranked: {q_data['reranked']}")
    print("-" * 50)
    print(f"Answer:\n{q_data['answer']}")
    print("-" * 50)
    print("Citations Provided:")
    for cit in q_data["citations"]:
        print(f"  * [Source {cit['source_index']}] {cit['document']} (Page {cit['page']}) Score: {cit['relevance_score']:.3f}")

    # --------------------------------------------------------------------------
    # 6. Automated RAG Evaluation Benchmark Suite
    # --------------------------------------------------------------------------
    print("\n[TEST 6] Running Automated RAG Evaluation Benchmark (/api/v1/evaluate)...")
    eval_resp = client.post("/api/v1/evaluate")
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    summary = eval_data["summary"]

    print("\n" + "=" * 50)
    print("        RAG BENCHMARK EVALUATION REPORT")
    print("=" * 50)
    print(f"Total Test Cases Evaluated : {summary['total_queries_evaluated']}")
    print(f"Hit Rate @ K               : {summary['hit_rate_at_k'] * 100:.1f}%")
    print(f"Mean Reciprocal Rank (MRR) : {summary['mean_reciprocal_rank_mrr']:.4f}")
    print(f"Average Context Relevance  : {summary['average_context_relevance'] * 100:.1f}%")
    print(f"Average Faithfulness       : {summary['average_faithfulness'] * 100:.1f}%")
    print(f"Overall Evaluation Status  : {summary['benchmark_status']}")
    print("=" * 50)

    print("\nIndividual Test Case Details:")
    for idx, case in enumerate(eval_data["cases"], start=1):
        status_icon = "PASS" if case['hit'] else "FAIL"
        print(f"  {idx}. [{status_icon}] Query: \"{case['query']}\"")
        print(f"     Rank of Hit: {case['hit_rank']} | Context Relevance: {case['context_relevance']} | Faithfulness: {case['faithfulness']}")

    print("\n[SUCCESS] All pipeline tests and evaluation benchmarks completed successfully!")


if __name__ == "__main__":
    run_all_tests()

