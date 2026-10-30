"""Vector store factory: local FAISS (single node) or Qdrant (multi-instance)."""
from __future__ import annotations

from knowledge_forge.config import get_settings
from knowledge_forge.vector_store.base import VectorStore


def get_vector_store() -> VectorStore:
    if get_settings().vector_store == "qdrant":
        from knowledge_forge.vector_store.qdrant_store import QdrantStore  # optional dependency

        return QdrantStore()
    from knowledge_forge.vector_store.local_store import LocalFaissStore

    return LocalFaissStore()


__all__ = ["VectorStore", "get_vector_store"]
