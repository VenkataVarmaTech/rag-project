"""
================================================================================
BARE-METAL RAG FROM SCRATCH (No Frameworks, No Vector DBs, Pure Python & NumPy)
================================================================================

This script demonstrates the 5 foundational stages of Retrieval-Augmented Generation:
1. INGESTION  : Read raw text from a document.
2. CHUNKING   : Split text into overlapping windows so context isn't severed mid-sentence.
3. EMBEDDING  : Convert text chunks into 384-dimensional mathematical vectors.
4. RETRIEVAL  : Compare a user's question vector against all chunk vectors using Cosine Similarity.
5. GENERATION : Build an augmented prompt containing only the top-k most relevant chunks.
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------------------------
# STAGE 1: INGESTION (Loading the raw knowledge base)
# ------------------------------------------------------------------------------
def load_document(file_path: str) -> str:
    """Reads and returns the content of a text document."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------------------
# STAGE 2: CHUNKING (Sliding Window with Overlap)
# ------------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 350, chunk_overlap: int = 50) -> list[str]:
    """
    Splits text into chunks of roughly `chunk_size` characters with an overlap.
    
    Why overlap?
    If a key fact spans across character 340 to 360, a strict cut at 350 destroys 
    the meaning. Overlap ensures that boundary information is preserved in both chunks.
    """
    # Simple, transparent paragraph/sentence-aware or character sliding-window chunking
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


# ------------------------------------------------------------------------------
# STAGE 3: EMBEDDING (Converting Text into High-Dimensional Vectors)
# ------------------------------------------------------------------------------
class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[*] Loading embedding model: {model_name}...")
        # Uses local cache; all-MiniLM-L6-v2 produces 384-dimensional vectors
        self.model = SentenceTransformer(model_name, local_files_only=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embeds a list of text strings into normalized vector representations.
        normalize_embeddings=True makes the magnitude of every vector = 1.0,
        which turns Cosine Similarity into a simple, ultra-fast Dot Product!
        """
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ------------------------------------------------------------------------------
# STAGE 4: RETRIEVAL (Cosine Similarity Search with NumPy)
# ------------------------------------------------------------------------------
class BareMetalVectorIndex:
    """
    This replaces Pinecone, ChromaDB, Weaviate, FAISS, etc.
    Under the hood, a vector database is fundamentally an index of vectors 
    paired with metadata (the original text), queried via distance metrics.
    """
    def __init__(self):
        self.chunks: list[str] = []
        self.vectors: np.ndarray | None = None

    def add(self, chunks: list[str], vectors: np.ndarray):
        self.chunks.extend(chunks)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Computes the cosine similarity between the query vector and all stored vectors.
        Formula:
            Cosine Similarity = (A . B) / (||A|| * ||B||)
        Since vectors are already normalized (||A|| = 1, ||B|| = 1), 
        Cosine Similarity simplifies to:
            A . B  (Matrix-vector dot product)
        """
        # query_vector shape: (384,)
        # self.vectors shape: (N, 384)
        # dot_products shape: (N,)
        similarities = np.dot(self.vectors, query_vector)

        # Get indices of the top-k highest similarity scores
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": int(idx),
                "score": float(similarities[idx]),
                "text": self.chunks[idx]
            })
        return results


# ------------------------------------------------------------------------------
# STAGE 5: AUGMENTED PROMPT GENERATION
# ------------------------------------------------------------------------------
def build_augmented_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Synthesizes the retrieved knowledge chunks into an enriched prompt for the LLM.
    """
    context_blocks = []
    for i, res in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[Source {i} | Similarity Score: {res['score']:.4f}]:\n{res['text']}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are an accurate, grounded assistant. Answer the user's question using ONLY the retrieved context below.
If the answer cannot be determined from the context, state: 'The provided documentation does not contain this information.'

---
RETRIEVED CONTEXT:
{context_str}
---

USER QUESTION: {query}

GROUNDED ANSWER:"""
    return prompt


# ------------------------------------------------------------------------------
# EXECUTION / DEMONSTRATION
# ------------------------------------------------------------------------------
def main():
    kb_path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.txt")
    print("\n" + "="*70)
    print("STEP 1: INGESTION")
    print("="*70)
    raw_text = load_document(kb_path)
    print(f"Loaded document from {kb_path} ({len(raw_text)} characters)")

    print("\n" + "="*70)
    print("STEP 2: CHUNKING")
    print("="*70)
    chunks = chunk_text(raw_text, chunk_size=350, chunk_overlap=60)
    print(f"Total chunks created: {len(chunks)}")
    print("\n[Preview of Chunk #0]:")
    print("-" * 40)
    print(chunks[0])
    print("-" * 40)
    print("\n[Preview of Chunk #1] (Notice the overlap with Chunk #0):")
    print("-" * 40)
    print(chunks[1])
    print("-" * 40)

    print("\n" + "="*70)
    print("STEP 3: EMBEDDING CHUNKS")
    print("="*70)
    embedder = LocalEmbedder()
    chunk_vectors = embedder.embed(chunks)
    print(f"Generated embeddings matrix shape: {chunk_vectors.shape}")
    print(f"-> {chunk_vectors.shape[0]} chunks, each represented as a {chunk_vectors.shape[1]}-dimensional vector.")
    print(f"Sample values from Chunk #0 vector (first 5 numbers): {np.round(chunk_vectors[0][:5], 4)}")

    print("\n" + "="*70)
    print("STEP 4: STORING IN BARE-METAL VECTOR INDEX")
    print("="*70)
    index = BareMetalVectorIndex()
    index.add(chunks, chunk_vectors)
    print("Index populated successfully.")

    # Let's run a test query!
    sample_queries = [
        "What is the maximum allowed temperature for the mixing chamber before the alarm triggers?",
        "What are the steps if a vacuum leak occurs in Stage 1?",
        "Who is the facility director and what is their access level?",
        "What is the price of a coffee at the cafeteria?"  # Out of domain query!
    ]

    print("\n" + "="*70)
    print("STEP 5: RETRIEVAL & AUGMENTED PROMPT SYNTHESIS")
    print("="*70)

    for q in sample_queries:
        print(f"\n>>> USER QUERY: \"{q}\"")
        q_vector = embedder.embed([q])[0]
        results = index.search(q_vector, top_k=2)

        print("\nTop Retrieved Chunks:")
        for r in results:
            print(f"  - [Chunk #{r['chunk_id']}] Similarity: {r['score']:.4f}")
            snippet = r['text'].replace('\n', ' ')
            if len(snippet) > 100:
                snippet = snippet[:100] + "..."
            print(f"    Excerpt: {snippet}")

        # Show the prompt that would be fed to the LLM
        prompt = build_augmented_prompt(q, results)
        print("\nConstructed Augmented Prompt (What the LLM receives):")
        print("~" * 60)
        print(prompt)
        print("~" * 60)


if __name__ == "__main__":
    main()

