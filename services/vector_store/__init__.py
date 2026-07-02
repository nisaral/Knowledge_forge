"""Vector store factory — local FAISS or Qdrant for multi-instance deployments."""
from config import VECTOR_STORE

from services.vector_store.local_store import LocalFaissStore
from services.vector_store.qdrant_store import QdrantStore


def get_vector_store():
    if VECTOR_STORE == "qdrant":
        return QdrantStore()
    return LocalFaissStore()