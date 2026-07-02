"""Qdrant vector store for horizontally scalable multi-instance deployments."""
import logging
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import EMBED_DIM, QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL
from services.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class QdrantStore(VectorStore):
    def __init__(self):
        self.collection = QDRANT_COLLECTION
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            timeout=30,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        logger.info("Creating Qdrant collection '%s'", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="source",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    def _source_filter(self, source_filter: str | None) -> qm.Filter | None:
        if not source_filter:
            return None
        return qm.Filter(
            must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source_filter))]
        )

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
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def semantic_search(
        self,
        query_vector: np.ndarray,
        source_filter: str | None,
        k: int,
    ) -> list[dict[str, Any]]:
        vec = query_vector.flatten()
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vec.tolist(),
            query_filter=self._source_filter(source_filter),
            limit=k,
            with_payload=True,
        )
        hits = []
        for point in results:
            payload = point.payload or {}
            hits.append({
                "id": str(point.id),
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "source_type": payload.get("source_type", "text"),
                "metadata": payload.get("metadata", {}),
                "score": float(point.score or 0.0),
            })
        return hits

    def scroll_chunks(self, source_filter: str | None = None) -> list[dict]:
        chunks = []
        offset = None
        query_filter = self._source_filter(source_filter)
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                chunks.append({
                    "id": str(record.id),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "source_type": payload.get("source_type", "text"),
                    "metadata": payload.get("metadata", {}),
                })
            if offset is None:
                break
        return chunks

    def scroll_with_vectors(self) -> list[tuple[dict, np.ndarray]]:
        pairs = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                payload = record.payload or {}
                chunk = {
                    "id": str(record.id),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "source_type": payload.get("source_type", "text"),
                    "metadata": payload.get("metadata", {}),
                }
                vec = np.array(record.vector, dtype="float32")
                pairs.append((chunk, vec))
            if offset is None:
                break
        return pairs

    def count(self) -> int:
        info = self.client.get_collection(self.collection)
        return info.points_count or 0

    def clear(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._ensure_collection()

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return False

    def stats(self) -> dict:
        chunks = self.scroll_chunks()
        sources = {c["source"] for c in chunks}
        return {
            "backend": "qdrant",
            "chunks": len(chunks),
            "vectors": self.count(),
            "sources": len(sources),
            "collection": self.collection,
            "url": QDRANT_URL,
        }