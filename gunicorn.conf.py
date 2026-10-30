"""Gunicorn production configuration (all values overridable via env vars)."""
import logging
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# The local FAISS store keeps the index in process memory and writes JSON files without
# cross-process locking, so more than one worker would diverge and overwrite each other.
# Multiple workers are only safe with VECTOR_STORE=qdrant (+ REDIS_URL for cache sync).
_shared_store = os.environ.get("VECTOR_STORE", "local").strip().lower() == "qdrant"
workers = int(os.environ.get("GUNICORN_WORKERS", "2" if _shared_store else "1"))
if not _shared_store and workers != 1:
    logging.getLogger("gunicorn.error").warning(
        "GUNICORN_WORKERS=%s ignored: VECTOR_STORE=local requires a single worker.", workers
    )
    workers = 1
threads = int(os.environ.get("GUNICORN_THREADS", "8"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))
worker_class = "gthread"
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
preload_app = True
