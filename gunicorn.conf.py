"""Gunicorn production configuration."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Multi-worker safe when VECTOR_STORE=qdrant + REDIS_URL are configured
_default_workers = "2" if os.environ.get("VECTOR_STORE") == "qdrant" else "1"
workers = int(os.environ.get("GUNICORN_WORKERS", _default_workers))
threads = int(os.environ.get("GUNICORN_THREADS", "8"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
keepalive = 5
worker_class = "gthread"
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
preload_app = True