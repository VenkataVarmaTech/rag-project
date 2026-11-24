"""
Dense Embedding Wrapper Module.
Uses SentenceTransformer with normalized vectors (384-dimensions).
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME


class DenseEmbedder:
    _instance = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME):
        # Singleton pattern to prevent re-allocating model in memory
        if cls._instance is None:
            cls._instance = super(DenseEmbedder, cls).__new__(cls)
            cls._instance._init_model(model_name)
        return cls._instance

    def _init_model(self, model_name: str):
        print(f"[*] Initializing Dense Embedder with model: {model_name}...")
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.dimension = 384

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of texts into normalized numpy float32 vectors.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D vector."""
        return self.embed_texts([query])[0]

