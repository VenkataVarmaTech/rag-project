"""
ChromaDB Vector Store Wrapper Module.
Provides persistent storage, vector indexing with HNSW cosine distance, and metadata filtering.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from src.config import CHROMA_PERSIST_DIR


class ChromaVectorStore:
    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR, collection_name: str = "rag_knowledge_base"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """
        Adds text chunks, their vector embeddings, and metadata into ChromaDB.
        """
        if not chunks:
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB using dense vector similarity.
        Returns a list of structured candidate dictionaries.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        query_args: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"]
        }
        if filter_metadata:
            query_args["where"] = filter_metadata

        results = self.collection.query(**query_args)

        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0] if results.get("documents") else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            distances = results["distances"][0] if results.get("distances") else []

            for i in range(len(ids)):
                # In Chroma with cosine space: distance = 1 - cosine_similarity
                distance = distances[i] if i < len(distances) else 1.0
                similarity = max(0.0, 1.0 - distance)

                formatted_results.append({
                    "chunk_id": ids[i],
                    "text": docs[i] if i < len(docs) else "",
                    "score": float(similarity),
                    "metadata": metas[i] if i < len(metas) else {},
                    "retrieval_source": "dense"
                })

        return formatted_results

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

