"""Gemini embedding helpers with retry, model fallback, and batch resilience."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

import google.generativeai as genai
import numpy as np

from knowledge_forge.config import get_settings
from knowledge_forge.llm import ensure_gemini_configured

logger = logging.getLogger(__name__)

_active_model: str | None = None

# text-embedding-004 was removed from the API; gemini-embedding-001 is the supported replacement.
_MODEL_FALLBACKS = [
    "models/gemini-embedding-001",
    "models/embedding-001",
    "models/text-embedding-004",
]


def _model_candidates() -> list[str]:
    seen = set()
    candidates = []
    for m in [get_settings().embed_model, *_MODEL_FALLBACKS]:
        if not m:
            continue
        name = m if m.startswith("models/") else f"models/{m}"
        if name not in seen:
            seen.add(name)
            candidates.append(name)
    return candidates


def _parse_embedding(raw) -> list[float]:
    """Normalize embedding payloads across Gemini SDK versions."""
    if isinstance(raw, dict):
        return raw.get("values") or raw.get("embedding") or []
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], (list, tuple)):
            return list(raw[0])
        return list(raw)
    return []


def _embed_kwargs(task_type: str) -> dict:
    """Build embed_content kwargs — gemini-embedding-001 supports task_type + output_dimensionality."""
    kwargs = {"output_dimensionality": get_settings().embed_dim}
    model = _active_model or ""
    if "gemini-embedding-2" not in model:
        kwargs["task_type"] = task_type
    return kwargs


def _format_for_embedding_2(text: str, task_type: str) -> str:
    """gemini-embedding-2 uses prompt prefixes instead of task_type."""
    if task_type == "retrieval_query":
        return f"task: search result | query: {text}"
    return f"title: none | text: {text}"


def _call_embed(model: str, content, task_type: str) -> list[float]:
    ensure_gemini_configured()
    if "gemini-embedding-2" in model:
        if isinstance(content, list):
            content = [_format_for_embedding_2(t, task_type) for t in content]
        else:
            content = _format_for_embedding_2(content, task_type)
        result = genai.embed_content(
            model=model,
            content=content,
            output_dimensionality=get_settings().embed_dim,
        )
    else:
        result = genai.embed_content(
            model=model,
            content=content,
            **_embed_kwargs(task_type),
        )
    raw = result.get("embedding", result)
    if isinstance(content, list) and raw and isinstance(raw[0], (int, float)):
        return _parse_embedding(raw)
    if isinstance(content, list):
        return _parse_embedding(raw[0]) if raw else []
    return _parse_embedding(raw)


def _resolve_active_model() -> str:
    global _active_model
    if _active_model:
        return _active_model

    ensure_gemini_configured()
    probe = "KnowledgeForge embedding probe"
    last_err = None
    for model in _model_candidates():
        try:
            vec = _call_embed(model, probe, "retrieval_document")
            if len(vec) != get_settings().embed_dim:
                raise ValueError(f"Expected {get_settings().embed_dim}-dim embedding, got {len(vec)} from {model}")
            _active_model = model
            logger.info("Using embedding model: %s (%d-dim)", model, get_settings().embed_dim)
            return model
        except Exception as exc:
            last_err = exc
            logger.warning("Embedding model %s unavailable: %s", model, exc)

    raise RuntimeError(
        f"No working embedding model found. Tried: {_model_candidates()}. Last error: {last_err}"
    )


def _embed_single(text: str, task_type: str) -> np.ndarray:
    global _active_model
    _resolve_active_model()
    last_err = None
    for attempt in range(get_settings().embed_max_retries):
        try:
            vec = _call_embed(_active_model, text, task_type)
            if len(vec) != get_settings().embed_dim:
                raise ValueError(f"Expected {get_settings().embed_dim}-dim embedding, got {len(vec)}")
            return np.array(vec, dtype="float32")
        except Exception as exc:
            last_err = exc
            if "404" in str(exc) or "not found" in str(exc).lower():
                _active_model = None
                _resolve_active_model()
            logger.warning("Embedding retry %d/%d: %s", attempt + 1, get_settings().embed_max_retries, exc)
            if attempt + 1 < get_settings().embed_max_retries:
                time.sleep(min(8, 2 ** attempt))
    raise RuntimeError(f"Embedding failed after {get_settings().embed_max_retries} attempts: {last_err}")


def gemini_embed(texts: list[str]) -> np.ndarray:
    """Batch-embed documents with per-item fallback on batch failure."""
    if not texts:
        return np.empty((0, get_settings().embed_dim), dtype="float32")

    _resolve_active_model()
    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), get_settings().embed_batch_size):
        batch = texts[i:i + get_settings().embed_batch_size]
        try:
            if "gemini-embedding-2" in (_active_model or ""):
                raise ValueError("Batch via per-item for embedding-2")
            ensure_gemini_configured()
            result = genai.embed_content(
                model=_active_model,
                content=batch,
                **_embed_kwargs("retrieval_document"),
            )
            raw = result.get("embedding", [])
            if batch and isinstance(raw[0], (int, float)):
                parsed = [_parse_embedding(raw)]
            else:
                parsed = [_parse_embedding(item) for item in raw]

            if len(parsed) != len(batch):
                raise ValueError(f"Batch size mismatch: sent {len(batch)}, got {len(parsed)}")

            for vec in parsed:
                if len(vec) != get_settings().embed_dim:
                    raise ValueError(f"Expected {get_settings().embed_dim}-dim embedding, got {len(vec)}")
                vectors.append(np.array(vec, dtype="float32"))
        except Exception as exc:
            logger.warning("Batch embed failed (%s); falling back to per-item.", exc)
            for text in batch:
                vectors.append(_embed_single(text, "retrieval_document"))

    return np.vstack(vectors)


_QUERY_CACHE_SIZE = 256
_query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_query_cache_lock = threading.Lock()


def gemini_embed_query(query: str) -> np.ndarray:
    """Embed a single query string (LRU-cached so repeated questions are free)."""
    key = f"{_active_model}|{query}"
    with _query_cache_lock:
        if key in _query_cache:
            _query_cache.move_to_end(key)
            return _query_cache[key].copy()
    vec = _embed_single(query, "retrieval_query").reshape(1, -1)
    with _query_cache_lock:
        _query_cache[key] = vec.copy()
        if len(_query_cache) > _QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)
    return vec


def reset_embedding_state() -> None:
    """Forget the probed model and cached query vectors (used by tests)."""
    global _active_model
    _active_model = None
    with _query_cache_lock:
        _query_cache.clear()
