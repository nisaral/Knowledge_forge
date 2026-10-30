"""Centralised configuration loaded from environment variables.

All settings are read once into an immutable :class:`Settings` object returned by
:func:`get_settings`. Tests (or long-running processes that change the environment)
can call ``get_settings.cache_clear()`` to re-read the environment.
"""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

logger = logging.getLogger(__name__)


def _int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw not in (None, "") else default
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %s", name, raw, default)
        value = default
    if minimum is not None and value < minimum:
        logger.warning("%s=%s is below minimum %s; clamping.", name, value, minimum)
        value = minimum
    return value


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return default


def _str_env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


@dataclass(frozen=True)
class Settings:
    # API keys & security
    gemini_api_key: str = ""
    flask_secret: str = ""
    cors_origins: tuple[str, ...] = ()

    # Models
    embed_model: str = "models/gemini-embedding-001"
    gen_model: str = "gemini-2.0-flash"
    embed_dim: int = 768

    # Chunking / retrieval
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k_coarse: int = 12
    top_k_final: int = 5
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    rerank: bool = True

    # Persistence
    data_dir: str = "data"

    # Limits & resilience
    max_upload_mb: int = 50
    max_text_chars: int = 2_000_000
    max_question_chars: int = 4000
    embed_batch_size: int = 100
    embed_max_retries: int = 3
    llm_max_retries: int = 3
    gemini_file_poll_seconds: int = 3
    gemini_file_max_polls: int = 40
    crawl_timeout_seconds: int = 15
    allow_private_urls: bool = False
    chat_history_turns: int = 6

    # FAISS
    faiss_ivf_threshold: int = 1000
    faiss_ivf_nlist: int = 64

    # Vector store
    vector_store: str = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledge_forge"

    # Redis (cross-instance cache invalidation)
    redis_url: str = ""
    kb_cache_version_key: str = "kf:kb:version"

    @property
    def kb_file(self) -> str:
        return os.path.join(self.data_dir, "knowledge_base.json")

    @property
    def index_file(self) -> str:
        return os.path.join(self.data_dir, "faiss_index.bin")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @classmethod
    def from_env(cls) -> Settings:
        if load_dotenv is not None:
            load_dotenv(override=False)

        chunk_size = _int_env("CHUNK_SIZE", 400, minimum=20)
        overlap = _int_env("CHUNK_OVERLAP", 50, minimum=0)
        if overlap >= chunk_size:
            logger.warning("CHUNK_OVERLAP (%s) >= CHUNK_SIZE (%s); using size//4.", overlap, chunk_size)
            overlap = chunk_size // 4

        sem_w = max(0.0, _float_env("SEMANTIC_WEIGHT", 0.7))
        bm_w = max(0.0, _float_env("BM25_WEIGHT", 0.3))
        if sem_w + bm_w == 0:
            sem_w, bm_w = 0.7, 0.3

        top_k_final = _int_env("TOP_K_FINAL", 5, minimum=1)
        top_k_coarse = max(top_k_final, _int_env("TOP_K_COARSE", 12, minimum=1))

        cors = tuple(o.strip() for o in _str_env("CORS_ORIGINS").split(",") if o.strip())
        vector_store = _str_env("VECTOR_STORE", "local").lower()
        if vector_store not in ("local", "qdrant"):
            logger.warning("Unknown VECTOR_STORE=%r; falling back to 'local'.", vector_store)
            vector_store = "local"

        return cls(
            gemini_api_key=_str_env("GEMINI_API_KEY") or _str_env("GOOGLE_API_KEY"),
            flask_secret=_str_env("FLASK_SECRET"),
            cors_origins=cors,
            embed_model=_str_env("EMBED_MODEL", "models/gemini-embedding-001"),
            gen_model=_str_env("GEN_MODEL", "gemini-2.0-flash"),
            embed_dim=_int_env("EMBED_DIM", 768, minimum=8),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            top_k_coarse=top_k_coarse,
            top_k_final=top_k_final,
            semantic_weight=sem_w,
            bm25_weight=bm_w,
            rerank=_str_env("RERANK", "true").lower() not in ("0", "false", "no", "off"),
            data_dir=_str_env("DATA_DIR", "data"),
            max_upload_mb=_int_env("MAX_UPLOAD_MB", 50, minimum=1),
            max_text_chars=_int_env("MAX_TEXT_CHARS", 2_000_000, minimum=1000),
            max_question_chars=_int_env("MAX_QUESTION_CHARS", 4000, minimum=50),
            embed_batch_size=_int_env("EMBED_BATCH_SIZE", 100, minimum=1),
            embed_max_retries=_int_env("EMBED_MAX_RETRIES", 3, minimum=1),
            llm_max_retries=_int_env("LLM_MAX_RETRIES", 3, minimum=1),
            gemini_file_poll_seconds=_int_env("GEMINI_FILE_POLL_SECONDS", 3, minimum=1),
            gemini_file_max_polls=_int_env("GEMINI_FILE_MAX_POLLS", 40, minimum=1),
            crawl_timeout_seconds=_int_env("CRAWL_TIMEOUT_SECONDS", 15, minimum=1),
            allow_private_urls=_str_env("ALLOW_PRIVATE_URLS", "false").lower() in ("1", "true", "yes"),
            chat_history_turns=_int_env("CHAT_HISTORY_TURNS", 6, minimum=0),
            faiss_ivf_threshold=_int_env("FAISS_IVF_THRESHOLD", 1000, minimum=10),
            faiss_ivf_nlist=_int_env("FAISS_IVF_NLIST", 64, minimum=1),
            vector_store=vector_store,
            qdrant_url=_str_env("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=_str_env("QDRANT_API_KEY"),
            qdrant_collection=_str_env("QDRANT_COLLECTION", "knowledge_forge"),
            redis_url=_str_env("REDIS_URL"),
            kb_cache_version_key=_str_env("KB_CACHE_VERSION_KEY", "kf:kb:version"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def flask_secret(settings: Settings) -> str:
    """Return the configured Flask secret or an ephemeral random one (with a warning)."""
    if settings.flask_secret:
        return settings.flask_secret
    logger.warning("FLASK_SECRET not set; using a random per-process secret.")
    return secrets.token_hex(32)
