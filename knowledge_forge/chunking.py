"""Text normalisation and chunking."""
from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip NUL/control characters."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return _WS.sub(" ", text).strip()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window word chunker.

    ``size`` is the window length in words and ``overlap`` the number of words shared
    between consecutive chunks. The final window is not emitted if it is fully contained
    in the previous one.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")

    words = normalize_text(text).split(" ")
    words = [w for w in words if w]
    if not words:
        return []

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def content_hash(text: str) -> str:
    """Stable hash used to deduplicate chunks and avoid re-embedding."""
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()[:32]
