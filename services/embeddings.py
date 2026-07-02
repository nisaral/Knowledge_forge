"""Gemini embedding helpers with retry and batch resilience."""
import logging
import time

import google.generativeai as genai
import numpy as np

from config import (
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MAX_RETRIES,
    EMBED_MODEL,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)

_configured = False


def ensure_gemini_configured() -> None:
    global _configured
    if _configured:
        return
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    _configured = True


def _parse_embedding(raw) -> list[float]:
    """Normalize embedding payloads across Gemini SDK versions."""
    if isinstance(raw, dict):
        return raw.get("values") or raw.get("embedding") or []
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], (list, tuple)):
            return list(raw[0])
        return list(raw)
    return []


def _embed_single(text: str, task_type: str) -> np.ndarray:
    ensure_gemini_configured()
    last_err = None
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type=task_type,
            )
            vec = _parse_embedding(result.get("embedding", result))
            if len(vec) != EMBED_DIM:
                raise ValueError(f"Expected {EMBED_DIM}-dim embedding, got {len(vec)}")
            return np.array(vec, dtype="float32")
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            logger.warning("Embedding retry %d/%d: %s", attempt + 1, EMBED_MAX_RETRIES, exc)
            time.sleep(wait)
    raise RuntimeError(f"Embedding failed after {EMBED_MAX_RETRIES} attempts: {last_err}")


def gemini_embed(texts: list[str]) -> np.ndarray:
    """Batch-embed documents with per-item fallback on batch failure."""
    if not texts:
        return np.empty((0, EMBED_DIM), dtype="float32")

    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        try:
            ensure_gemini_configured()
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=batch,
                task_type="retrieval_document",
            )
            raw = result.get("embedding", [])
            if batch and isinstance(raw[0], (int, float)):
                # Single-item response shape
                parsed = [_parse_embedding(raw)]
            else:
                parsed = [_parse_embedding(item) for item in raw]

            if len(parsed) != len(batch):
                raise ValueError(f"Batch size mismatch: sent {len(batch)}, got {len(parsed)}")

            for vec in parsed:
                if len(vec) != EMBED_DIM:
                    raise ValueError(f"Expected {EMBED_DIM}-dim embedding, got {len(vec)}")
                vectors.append(np.array(vec, dtype="float32"))
        except Exception as exc:
            logger.warning("Batch embed failed (%s); falling back to per-item.", exc)
            for text in batch:
                vectors.append(_embed_single(text, "retrieval_document"))

    return np.vstack(vectors)


def gemini_embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    vec = _embed_single(query, "retrieval_query")
    return vec.reshape(1, -1)