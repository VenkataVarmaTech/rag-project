"""
Central Configuration for the Enterprise RAG System.
"""

import os
from pathlib import Path

# Disable ChromaDB telemetry and force offline mode to eliminate socket connection timeouts
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_db"

# Ensure necessary directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Default Model Configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking Parameters
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80

# Retrieval Defaults
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_RERANK_TOP_K = 3
RRF_K_PARAMETER = 60  # Standard constant used in Reciprocal Rank Fusion

# Server Configurations
API_HOST = "127.0.0.1"
API_PORT = 8000

