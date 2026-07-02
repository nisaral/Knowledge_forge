"""Centralized configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ─── API Keys & Security ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FLASK_SECRET = os.environ.get("FLASK_SECRET", "kf-v2-secret-change-me")

# ─── Model Configuration ─────────────────────────────────────────────────────
EMBED_MODEL = os.environ.get("EMBED_MODEL", "models/text-embedding-004")
GEN_MODEL = os.environ.get("GEN_MODEL", "gemini-2.0-flash")
EMBED_DIM = _int_env("EMBED_DIM", 768)

# ─── RAG / Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = _int_env("CHUNK_SIZE", 400)
CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 50)
TOP_K_COARSE = _int_env("TOP_K_COARSE", 12)
TOP_K_FINAL = _int_env("TOP_K_FINAL", 5)
SEMANTIC_WEIGHT = _float_env("SEMANTIC_WEIGHT", 0.7)
BM25_WEIGHT = _float_env("BM25_WEIGHT", 0.3)

# ─── Persistence ─────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", ".")
KB_FILE = os.path.join(DATA_DIR, "knowledge_base.json")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

# ─── Limits & Resilience ─────────────────────────────────────────────────────
MAX_UPLOAD_MB = _int_env("MAX_UPLOAD_MB", 50)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
EMBED_BATCH_SIZE = _int_env("EMBED_BATCH_SIZE", 100)
EMBED_MAX_RETRIES = _int_env("EMBED_MAX_RETRIES", 3)
LLM_MAX_RETRIES = _int_env("LLM_MAX_RETRIES", 3)
GEMINI_FILE_POLL_SECONDS = _int_env("GEMINI_FILE_POLL_SECONDS", 3)
GEMINI_FILE_MAX_POLLS = _int_env("GEMINI_FILE_MAX_POLLS", 40)
CRAWL_TIMEOUT_SECONDS = _int_env("CRAWL_TIMEOUT_SECONDS", 15)
CHAT_HISTORY_TURNS = _int_env("CHAT_HISTORY_TURNS", 6)

# FAISS IVF kicks in when chunk count exceeds this threshold (local mode only)
FAISS_IVF_THRESHOLD = _int_env("FAISS_IVF_THRESHOLD", 1000)
FAISS_IVF_NLIST = _int_env("FAISS_IVF_NLIST", 64)

# ─── Vector Store (local = FAISS, qdrant = multi-instance) ───────────────────
VECTOR_STORE = os.environ.get("VECTOR_STORE", "local").lower()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "knowledge_forge")

# ─── Redis (sessions + cross-instance BM25 cache sync) ───────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "")
KB_CACHE_VERSION_KEY = os.environ.get("KB_CACHE_VERSION_KEY", "kf:kb:version")
SESSION_TYPE = os.environ.get("SESSION_TYPE", "redis" if REDIS_URL else "filesystem")