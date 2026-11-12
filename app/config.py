import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "voicerag_docs")
DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 384  # output size of bge-small-en-v1.5
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "250"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))