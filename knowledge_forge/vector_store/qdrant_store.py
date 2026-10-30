"""Qdrant vector store for horizontally scalable multi-instance deployments."""
from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from knowledge_forge.config import get_settings
from knowledge_forge.errors import ConfigurationError
from knowledge_forge.vector_store.base import VectorStore

logger = logging.getLogger(__name__)

_SCROLL_PAGE = 256


def _payload_to_chunk(point_id, payload: dict | None) -> dict:
    payload = payload or {}
    return {
        "id": str(point_id),
        "text": payload.get("text", ""),
        "source": payload.get("source", ""),
        "source_type": payload.get("source_type", "text"),
        "metadata": payload.get("metadata", {}),
        "hash": payload.get("hash", ""),
    }


class QdrantStore(VectorStore):
    def __init__(self):
        settings = get_settings()
        self.dim = settings.embed_dim
        self.collection = settings.qdrant_collection
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout_seconds,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self._check_dimension()
            return
        logger.info("Creating Qdrant collection '%s'", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="source",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    def _check_dimension(self) -> None:
        try:
            vectors = self.client.get_collection(self.collection).config.params.vectors
            size = getattr(vectors, "size", None)
        except Exception:  # pragma: no cover - informational only
            return
        if size is not None and int(size) != self.dim:
            raise ConfigurationError(
                f"Qdrant collection '{self.collection}' stores {size}-dim vectors but EMBED_DIM is "
                f"{self.dim}. Use a new QDRANT_COLLECTION or set EMBED_DIM={size}."
            )

    @staticmethod
    def _source_filter(source_filter: str | None) -> qm.Filter | None:
        if not source_filter:
            return None
        return qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source_filter))])

    def upsert_chunks(self, records: list[dict], vectors: np.ndarray) -> None:
        points = []
        for record, vec in zip(records, vectors):
            point_id = record.get("id") or str(uuid.uuid4())
            record["id"] = point_id
            normalized = vec / (np.linalg.norm(vec) + 1e-12)
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=normalized.tolist(),
                    payload={
                        "text": record["text"],
                        "source": record["source"],
                        "source_type": record.get("source_type", "text"),
                        "metadata": record.get("metadata", {}),
                        "hash": record.get("hash", ""),
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def semantic_search(self, query_vector: np.ndarray, source_filter: str | None, k: int) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        vec = np.asarray(query_vector, dtype="float32").flatten()
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        results = self.client.query_points(
            collection_name=self.collection,
            query=vec.tolist(),
            query_filter=self._source_filter(source_filter),
            limit=k,
            with_payload=True,
        ).points
        return [{**_payload_to_chunk(p.id, p.payload), "score": float(p.score or 0.0)} for p in results]

    def _scroll(self, query_filter, with_vectors: bool):
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=_SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            yield from records
            if offset is None:
                break

    def scroll_chunks(self, source_filter: str | None = None) -> list[dict]:
        return [_payload_to_chunk(r.id, r.payload) for r in self._scroll(self._source_filter(source_filter), False)]

    def scroll_with_vectors(self) -> list[tuple[dict, np.ndarray]]:
        return [
            (_payload_to_chunk(r.id, r.payload), np.array(r.vector, dtype="float32"))
            for r in self._scroll(None, True)
        ]

    def delete_source(self, source: str) -> int:
        if not source:  # an empty filter would match (and delete) every point
            return 0
        flt = self._source_filter(source)
        removed = self.client.count(collection_name=self.collection, count_filter=flt, exact=True).count
        if removed:
            self.client.delete(
                collection_name=self.collection, points_selector=qm.FilterSelector(filter=flt), wait=True
            )
        return int(removed)

    def delete_ids(self, ids: list[str]) -> int:
        ids = [i for i in ids if i]
        if not ids:
            return 0
        self.client.delete(collection_name=self.collection, points_selector=qm.PointIdsList(points=ids), wait=True)
        return len(ids)

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def clear(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._ensure_collection()

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", type(exc).__name__)
            return False

    def stats(self) -> dict:
        n = self.count()
        return {"backend": "qdrant", "chunks": n, "vectors": n, "collection": self.collection}
