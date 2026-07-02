"""Vector store interface for KnowledgeForge."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def upsert_chunks(self, records: list[dict], vectors: np.ndarray) -> None:
        """Insert or update chunk records with their embedding vectors."""

    @abstractmethod
    def semantic_search(
        self,
        query_vector: np.ndarray,
        source_filter: str | None,
        k: int,
    ) -> list[dict[str, Any]]:
        """Return ranked hits: {id, text, source, source_type, metadata, score}."""

    @abstractmethod
    def scroll_chunks(self, source_filter: str | None = None) -> list[dict]:
        """Return all chunk records (optionally filtered by source)."""

    @abstractmethod
    def scroll_with_vectors(self) -> list[tuple[dict, np.ndarray]]:
        """Return all chunks paired with vectors (for clustering)."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored vectors."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored data."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the backing store is reachable."""

    @abstractmethod
    def stats(self) -> dict:
        """Return store-specific statistics."""