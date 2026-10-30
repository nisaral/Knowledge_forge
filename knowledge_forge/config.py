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


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
RENDER_JS_MODES = ("auto", "always", "never")


def _bool_env(name: str, default: bool) -> bool:
    raw = _str_env(name).lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    if raw:
        logger.warning("Invalid boolean for %s=%r; using default %s", name, raw, default)
    return default


def _list_env(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in _str_env(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # API keys & security
    gemini_api_key: str = ""
    flask_secret: str = ""
    cors_origins: tuple[str, ...] = ()

    # Models
    embed_model: str = "models/gemini-embedding-001"
    embed_fallback_models: tuple[str, ...] = ("models/gemini-embedding-001", "models/text-embedding-004")
    gen_model: str = "gemini-2.0-flash"
    embed_dim: int = 768
    llm_temperature: float = 0.3
    llm_max_output_tokens: int = 0  # 0 = model default
    gemini_timeout_seconds: int = 60
    llm_retry_base_seconds: float = 1.0
    llm_retry_max_seconds: float = 8.0

    # Chunking / retrieval
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k_coarse: int = 12
    top_k_final: int = 5
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    rerank: bool = True
    min_relevance: float = 0.0
    citation_snippet_chars: int = 320
    history_message_chars: int = 2000
    summary_max_chars: int = 24_000
    entity_max_chars: int = 6000
    tool_topic_max_chars: int = 300
    source_name_max_chars: int = 500
    cluster_min_chunks: int = 5
    cluster_max_topics: int = 5

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
    crawl_max_bytes: int = 5 * 1024 * 1024
    crawl_max_redirects: int = 5
    crawl_min_chars: int = 50
    crawl_respect_robots: bool = True
    crawl_user_agent: str = "KnowledgeForge/2 (+https://github.com/nisaral/Knowledge_forge)"
    allow_private_urls: bool = False
    # Headless-browser rendering for JavaScript pages (needs the optional "js" extra)
    render_js: str = "never"  # auto | always | never
    render_js_timeout_seconds: int = 20
    render_js_max_concurrency: int = 2
    render_js_max_requests: int = 150
    render_js_settle_ms: int = 1500
    render_js_min_chars: int = 500
    render_js_browser_path: str = ""
    pdf_ocr_min_chars: int = 100
    youtube_languages: tuple[str, ...] = ("en", "en-US", "en-GB")
    youtube_metadata_fallback: bool = False
    chat_history_turns: int = 6
    rate_limit_per_minute: int = 30
    trusted_proxies: int = 0

    # Server / logging
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "text"

    # FAISS
    faiss_ivf_threshold: int = 1000
    faiss_ivf_nlist: int = 64
    faiss_nprobe: int = 16

    # Vector store
    vector_store: str = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledge_forge"
    qdrant_timeout_seconds: int = 30

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

        cors = _list_env("CORS_ORIGINS")
        if "*" in cors:
            logger.warning("CORS_ORIGINS contains '*': any website can call this API from a browser.")
        log_format = _str_env("LOG_FORMAT", "text").lower()
        if log_format not in ("text", "json"):
            log_format = "text"
        render_js = _str_env("RENDER_JS", "never").lower()
        if render_js not in RENDER_JS_MODES:
            logger.warning("Invalid RENDER_JS=%r; using 'never'", render_js)
            render_js = "never"
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
            embed_fallback_models=_list_env(
                "EMBED_FALLBACK_MODELS", "models/gemini-embedding-001,models/text-embedding-004"
            ),
            embed_dim=_int_env("EMBED_DIM", 768, minimum=8),
            llm_temperature=min(2.0, max(0.0, _float_env("LLM_TEMPERATURE", 0.3))),
            llm_max_output_tokens=_int_env("LLM_MAX_OUTPUT_TOKENS", 0, minimum=0),
            gemini_timeout_seconds=_int_env("GEMINI_TIMEOUT_SECONDS", 60, minimum=5),
            llm_retry_base_seconds=max(0.0, _float_env("LLM_RETRY_BASE_SECONDS", 1.0)),
            llm_retry_max_seconds=max(0.0, _float_env("LLM_RETRY_MAX_SECONDS", 8.0)),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            top_k_coarse=top_k_coarse,
            top_k_final=top_k_final,
            semantic_weight=sem_w,
            bm25_weight=bm_w,
            rerank=_bool_env("RERANK", True),
            min_relevance=max(0.0, _float_env("MIN_RELEVANCE", 0.0)),
            citation_snippet_chars=_int_env("CITATION_SNIPPET_CHARS", 320, minimum=40),
            history_message_chars=_int_env("HISTORY_MESSAGE_CHARS", 2000, minimum=100),
            summary_max_chars=_int_env("SUMMARY_MAX_CHARS", 24_000, minimum=1000),
            entity_max_chars=_int_env("ENTITY_MAX_CHARS", 6000, minimum=500),
            tool_topic_max_chars=_int_env("TOOL_TOPIC_MAX_CHARS", 300, minimum=20),
            source_name_max_chars=_int_env("SOURCE_NAME_MAX_CHARS", 500, minimum=40),
            cluster_min_chunks=_int_env("CLUSTER_MIN_CHUNKS", 5, minimum=3),
            cluster_max_topics=_int_env("CLUSTER_MAX_TOPICS", 5, minimum=2),
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
            crawl_max_bytes=_int_env("CRAWL_MAX_BYTES", 5 * 1024 * 1024, minimum=10_000),
            crawl_max_redirects=_int_env("CRAWL_MAX_REDIRECTS", 5, minimum=0),
            crawl_min_chars=_int_env("CRAWL_MIN_CHARS", 50, minimum=1),
            crawl_respect_robots=_bool_env("CRAWL_RESPECT_ROBOTS", True),
            crawl_user_agent=_str_env(
                "CRAWL_USER_AGENT", "KnowledgeForge/2 (+https://github.com/nisaral/Knowledge_forge)"
            ),
            allow_private_urls=_bool_env("ALLOW_PRIVATE_URLS", False),
            render_js=render_js,
            render_js_timeout_seconds=_int_env("RENDER_JS_TIMEOUT_SECONDS", 20, minimum=2),
            render_js_max_concurrency=_int_env("RENDER_JS_MAX_CONCURRENCY", 2, minimum=1),
            render_js_max_requests=_int_env("RENDER_JS_MAX_REQUESTS", 150, minimum=1),
            render_js_settle_ms=_int_env("RENDER_JS_SETTLE_MS", 1500, minimum=0),
            render_js_min_chars=_int_env("RENDER_JS_MIN_CHARS", 500, minimum=1),
            render_js_browser_path=_str_env("RENDER_JS_BROWSER_PATH"),
            pdf_ocr_min_chars=_int_env("PDF_OCR_MIN_CHARS", 100, minimum=0),
            youtube_languages=_list_env("YOUTUBE_LANGUAGES", "en,en-US,en-GB"),
            youtube_metadata_fallback=_bool_env("YOUTUBE_METADATA_FALLBACK", False),
            chat_history_turns=_int_env("CHAT_HISTORY_TURNS", 6, minimum=0),
            rate_limit_per_minute=_int_env("RATE_LIMIT_PER_MINUTE", 30, minimum=0),
            trusted_proxies=_int_env("TRUSTED_PROXIES", 0, minimum=0),
            host=_str_env("HOST", "127.0.0.1"),
            port=_int_env("PORT", 5000, minimum=1),
            debug=_bool_env("FLASK_DEBUG", False),
            log_level=_str_env("LOG_LEVEL", "INFO").upper(),
            log_format=log_format,
            faiss_ivf_threshold=_int_env("FAISS_IVF_THRESHOLD", 1000, minimum=10),
            faiss_ivf_nlist=_int_env("FAISS_IVF_NLIST", 64, minimum=1),
            faiss_nprobe=_int_env("FAISS_NPROBE", 16, minimum=1),
            vector_store=vector_store,
            qdrant_url=_str_env("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=_str_env("QDRANT_API_KEY"),
            qdrant_collection=_str_env("QDRANT_COLLECTION", "knowledge_forge"),
            qdrant_timeout_seconds=_int_env("QDRANT_TIMEOUT_SECONDS", 30, minimum=1),
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
