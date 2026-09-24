import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "llama3:8b")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "768"))

# ChromaDB
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
CHROMA_COLLECTION_NAME = "rag_chunks"

# BM25
BM25_INDEX_PATH = str(PROJECT_ROOT / "bm25_index.pkl")

# Chunking defaults
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEMANTIC_SIMILARITY_THRESHOLD = 0.5

# Deduplication
DEDUP_SIMILARITY_THRESHOLD = 0.95

# Retrieval
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
FUSION_TOP_K = 20
RERANK_TOP_K = 5
RRF_DENSE_WEIGHT = 0.7
RRF_SPARSE_WEIGHT = 0.3

# Dataset
DATASET_DIR = str(PROJECT_ROOT / "data" / "EnterpriseRAG-Bench" / "data")
