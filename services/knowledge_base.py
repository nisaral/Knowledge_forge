"""Thread-safe hybrid RAG knowledge base with pluggable vector backends."""
import json
import logging
import threading
import uuid

import numpy as np
from rank_bm25 import BM25Okapi

from config import BM25_WEIGHT, SEMANTIC_WEIGHT, TOP_K_COARSE, TOP_K_FINAL
from services.cache import bump_kb_version, get_kb_version, reset_kb_version
from services.embeddings import gemini_embed, gemini_embed_query
from services.llm_utils import clean_llm_output, generate_content
from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self):
        self._lock = threading.RLock()
        self._store = get_vector_store()
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[dict] = []
        self._bm25_pool_len = -1
        self._cache_version = -1

    # ─── Cache Sync (multi-instance BM25) ────────────────────────────────────

    def _sync_bm25_cache(self, source_filter: str | None = None) -> list[dict]:
        version = get_kb_version()
        if version != self._cache_version or not self._bm25_chunks:
            self._bm25_chunks = self._store.scroll_chunks()
            self._cache_version = version
            self._bm25 = None

        if source_filter:
            return [c for c in self._bm25_chunks if c["source"] == source_filter]
        return self._bm25_chunks

    def _get_bm25(self, pool: list[dict]) -> BM25Okapi:
        if self._bm25 is None or self._bm25_pool_len != len(pool):
            tokenized = [c["text"].lower().split() for c in pool]
            self._bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([["placeholder"]])
            self._bm25_pool_len = len(pool)
        return self._bm25

    def _invalidate_local_cache(self) -> None:
        self._bm25 = None
        self._bm25_chunks = []
        self._bm25_pool_len = -1
        self._cache_version = -1

    # ─── Public API ──────────────────────────────────────────────────────────

    @property
    def chunks(self) -> list[dict]:
        """Backward-compatible chunk access."""
        with self._lock:
            return self._sync_bm25_cache()

    def count(self) -> int:
        with self._lock:
            return self._store.count()

    def list_chunks(self, source: str | None = None) -> list[dict]:
        with self._lock:
            return self._sync_bm25_cache(source)

    def load(self) -> None:
        with self._lock:
            self._invalidate_local_cache()
            self._sync_bm25_cache()

    def save(self) -> None:
        if hasattr(self._store, "save"):
            with self._lock:
                self._store.save()

    def clear(self, save: bool = True) -> None:
        with self._lock:
            if save:
                self._store.clear()
            else:
                if hasattr(self._store, "chunks"):
                    self._store.chunks = []
                    self._store.index = self._store._new_flat_index()
                    self._store._use_ivf = False
                    self._store._id_to_idx = {}
                else:
                    self._store.clear()
            reset_kb_version()
            bump_kb_version()
            self._invalidate_local_cache()
            logger.info("Knowledge base cleared.")

    def add_chunks(
        self,
        chunks: list[str],
        source: str,
        source_type: str,
        metadata: dict | None = None,
    ) -> bool:
        if not chunks:
            return False

        with self._lock:
            vectors = gemini_embed(chunks)
            records = [
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": source,
                    "source_type": source_type,
                    "metadata": metadata or {},
                }
                for chunk in chunks
            ]
            self._store.upsert_chunks(records, vectors)
            bump_kb_version()
            self._invalidate_local_cache()
            return True

    def hybrid_search(
        self,
        query: str,
        source_filter: str | None = None,
        k: int = TOP_K_FINAL,
    ) -> list[str]:
        with self._lock:
            if self._store.count() == 0:
                return []

            pool = self._sync_bm25_cache(source_filter)
            if not pool:
                return []

            q_vec = gemini_embed_query(query)
            sem_hits = self._store.semantic_search(q_vec, source_filter, TOP_K_COARSE)
            sem_scores = {h["id"]: h["score"] for h in sem_hits}

            bm25 = self._get_bm25(pool)
            bm25_scores_raw = bm25.get_scores(query.lower().split())
            bm25_max = max(bm25_scores_raw) if max(bm25_scores_raw) > 0 else 1.0
            bm25_scores = {
                pool[i]["id"]: bm25_scores_raw[i] / bm25_max
                for i in range(len(pool))
            }

            candidate_ids = set(sem_scores) | set(bm25_scores)
            combined = {}
            for cid in candidate_ids:
                combined[cid] = (
                    sem_scores.get(cid, 0.0) * SEMANTIC_WEIGHT
                    + bm25_scores.get(cid, 0.0) * BM25_WEIGHT
                )

            ranked_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:TOP_K_COARSE]
            id_to_chunk = {c["id"]: c for c in pool}
            top_chunks = [id_to_chunk[cid]["text"] for cid, _ in ranked_ids if cid in id_to_chunk]

            if len(top_chunks) > k:
                numbered = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(top_chunks))
                rerank_prompt = (
                    f"Given the query: '{query}'\n\n"
                    f"Here are {len(top_chunks)} candidate passages:\n{numbered}\n\n"
                    f"Return ONLY a JSON array of the {k} passage numbers (1-indexed) "
                    f"most relevant to the query, ordered by relevance. Example: [2,5,1,3,4]"
                )
                try:
                    resp_text = generate_content(rerank_prompt)
                    nums = json.loads(clean_llm_output(resp_text, "json"))
                    top_chunks = [top_chunks[n - 1] for n in nums if 1 <= n <= len(top_chunks)]
                except Exception as exc:
                    logger.warning("Re-rank failed, using coarse results: %s", exc)
                    top_chunks = top_chunks[:k]

            return top_chunks[:k]

    def get_embeddings_for_clustering(self) -> tuple[list[dict], np.ndarray]:
        with self._lock:
            pairs = self._store.scroll_with_vectors()
            if not pairs:
                return [], np.empty((0, 0), dtype="float32")
            chunks, vectors = zip(*pairs)
            return list(chunks), np.array(vectors, dtype="float32")

    def stats(self) -> dict:
        with self._lock:
            store_stats = self._store.stats()
            store_stats["cache_version"] = get_kb_version()
            return store_stats

    def health_check(self) -> bool:
        return self._store.health_check()