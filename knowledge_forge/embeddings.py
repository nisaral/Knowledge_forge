"""Gemini embedding helpers with retry, model fallback and batch resilience."""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

from knowledge_forge import gemini
from knowledge_forge.config import get_settings
from knowledge_forge.errors import ConfigurationError, KnowledgeForgeError, LLMError, RateLimitError
from knowledge_forge.llm import ensure_gemini_configured

logger = logging.getLogger(__name__)

_active_model: str | None = None
_model_lock = threading.Lock()

DOC_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"


def _model_candidates() -> list[str]:
    settings = get_settings()
    seen: set[str] = set()
    out: list[str] = []
    for m in (settings.embed_model, *settings.embed_fallback_models):
        if not m:
            continue
        name = m if m.startswith("models/") else f"models/{m}"
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _uses_prefixes(model: str) -> bool:
    """gemini-embedding-2 takes task prefixes in the text instead of ``task_type``."""
    return "gemini-embedding-2" in model


def _prepare(texts: list[str], model: str, task: str) -> tuple[list[str], str | None]:
    if not _uses_prefixes(model):
        return texts, task
    if task == QUERY_TASK:
        return [f"task: search result | query: {t}" for t in texts], None
    return [f"title: none | text: {t}" for t in texts], None


def _embed_raw(texts: list[str], model: str, task: str) -> np.ndarray:
    dim = get_settings().embed_dim
    prepared, task_type = _prepare(texts, model, task)
    vectors = gemini.embed(prepared, model, task_type, dim)
    if len(vectors) != len(texts):
        raise LLMError(f"Embedding count mismatch: sent {len(texts)}, got {len(vectors)}")
    for v in vectors:
        if len(v) != dim:
            raise LLMError(f"Expected {dim}-dim embeddings, got {len(v)} (check EMBED_DIM)")
    return np.asarray(vectors, dtype="float32")


def _resolve_active_model() -> str:
    """Probe candidate models once (thread-safe) and remember the first that works."""
    global _active_model
    with _model_lock:
        if _active_model:
            return _active_model
        ensure_gemini_configured()
        last: Exception | None = None
        for model in _model_candidates():
            try:
                _embed_raw(["KnowledgeForge embedding probe"], model, DOC_TASK)
                _active_model = model
                logger.info("Using embedding model %s (%d-dim)", model, get_settings().embed_dim)
                return model
            except (ConfigurationError, RateLimitError):
                raise
            except Exception as exc:
                last = exc
                logger.warning("Embedding model %s unavailable: %s", model, type(exc).__name__)
        if isinstance(last, KnowledgeForgeError):
            raise last
        raise gemini.translate_error(last, "embedding") if last else LLMError("No embedding model configured.")


def _embed_with_retry(texts: list[str], task: str) -> np.ndarray:
    model = _resolve_active_model()
    return gemini.with_retries(lambda: _embed_raw(texts, model, task), "embedding", get_settings().embed_max_retries)


def gemini_embed(texts: list[str]) -> np.ndarray:
    """Batch-embed documents, falling back to per-item calls when a batch fails."""
    dim = get_settings().embed_dim
    if not texts:
        return np.empty((0, dim), dtype="float32")
    batch_size = get_settings().embed_batch_size
    parts: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            parts.append(_embed_with_retry(batch, DOC_TASK))
        except (ConfigurationError, RateLimitError):
            raise
        except Exception as exc:
            if len(batch) == 1:
                raise
            logger.warning("Batch embed failed (%s); falling back to per-item.", type(exc).__name__)
            parts.extend(_embed_with_retry([t], DOC_TASK) for t in batch)
    return np.vstack(parts)


_QUERY_CACHE_SIZE = 256
_query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_query_cache_lock = threading.Lock()


def gemini_embed_query(query: str) -> np.ndarray:
    """Embed a query (LRU-cached per model so repeated questions are free)."""
    model = _resolve_active_model()
    key = f"{model}|{query}"
    with _query_cache_lock:
        if key in _query_cache:
            _query_cache.move_to_end(key)
            return _query_cache[key].copy()
    vec = _embed_with_retry([query], QUERY_TASK)[0]
    with _query_cache_lock:
        _query_cache[key] = vec.copy()
        while len(_query_cache) > _QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)
    return vec


def reset_embedding_state() -> None:
    """Forget the probed model and cached query vectors (used by tests)."""
    global _active_model
    with _model_lock:
        _active_model = None
    with _query_cache_lock:
        _query_cache.clear()
