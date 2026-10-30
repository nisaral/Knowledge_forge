"""Thread-safe hybrid (BM25 + semantic) knowledge base with optional LLM re-ranking."""
from __future__ import annotations

import logging
import re
import threading
import uuid

import numpy as np
from rank_bm25 import BM25Okapi

from knowledge_forge.cache import bump_kb_version, get_kb_version, reset_kb_version
from knowledge_forge.chunking import content_hash
from knowledge_forge.config import get_settings
from knowledge_forge.embeddings import gemini_embed, gemini_embed_query
from knowledge_forge.llm import generate_content, parse_rerank
from knowledge_forge.prompts import rerank_prompt
from knowledge_forge.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def fuse_scores(
    semantic: dict[str, float],
    keyword: dict[str, float],
    semantic_weight: float,
    keyword_weight: float,
) -> list[tuple[str, float]]:
    """Weighted score fusion. Keyword scores are expected to be normalised to [0, 1]."""
    ids = set(semantic) | set(keyword)
    fused = {
        cid: semantic.get(cid, 0.0) * semantic_weight + keyword.get(cid, 0.0) * keyword_weight
        for cid in ids
    }
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class KnowledgeBase:
    def __init__(self, store: VectorStore | None = None) -> None:
        self._lock = threading.RLock()
        self._store = store if store is not None else get_vector_store()
        self._chunks_cache: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._bm25_key: tuple | None = None
        self._cache_version = -1

    # ─── Cache handling ─────────────────────────────────────────────────────
    def _all_chunks(self) -> list[dict]:
        version = get_kb_version()
        if version != self._cache_version:
            self._chunks_cache = self._store.scroll_chunks()
            self._cache_version = version
            self._bm25 = None
            self._bm25_key = None
        return self._chunks_cache

    def _invalidate(self) -> None:
        bump_kb_version()
        self._cache_version = -1
        self._bm25 = None
        self._bm25_key = None

    def _bm25_for(self, pool: list[dict], source_filter: str | None) -> BM25Okapi:
        key = (self._cache_version, source_filter, len(pool))
        if self._bm25 is None or self._bm25_key != key:
            self._bm25 = BM25Okapi([tokenize(c["text"]) or ["_"] for c in pool])
            self._bm25_key = key
        return self._bm25

    # ─── Read API ───────────────────────────────────────────────────────────
    def count(self) -> int:
        with self._lock:
            return self._store.count()

    def list_chunks(self, source: str | None = None) -> list[dict]:
        with self._lock:
            chunks = self._all_chunks()
            return [c for c in chunks if c["source"] == source] if source else list(chunks)

    def list_sources(self) -> list[dict]:
        with self._lock:
            sources: dict[str, dict] = {}
            for c in self._all_chunks():
                entry = sources.setdefault(
                    c["source"],
                    {"source": c["source"], "type": c.get("source_type", "text"), "chunks": 0},
                )
                entry["chunks"] += 1
            return list(sources.values())

    def stats(self) -> dict:
        with self._lock:
            stats = self._store.stats()
            stats["cache_version"] = get_kb_version()
            return stats

    def health_check(self) -> bool:
        return self._store.health_check()

    # ─── Write API ──────────────────────────────────────────────────────────
    def add_chunks(
        self,
        chunks: list[str],
        source: str,
        source_type: str,
        metadata: dict | None = None,
    ) -> int:
        """Embed and store ``chunks``. Returns how many new chunks were stored.

        Chunks whose content hash already exists for the same source are skipped, so
        re-adding a document does not re-embed (or duplicate) it.
        """
        if not chunks:
            return 0
        with self._lock:
            existing = {c.get("hash") for c in self._all_chunks() if c["source"] == source}
        seen: set[str] = set()
        fresh: list[tuple[str, str]] = []
        for chunk in chunks:
            h = content_hash(chunk)
            if h in existing or h in seen:
                continue
            seen.add(h)
            fresh.append((chunk, h))
        if not fresh:
            return 0

        # Network-bound embedding happens outside the lock so reads are not blocked.
        vectors = gemini_embed([c for c, _ in fresh])
        records = [
            {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "hash": h,
                "source": source,
                "source_type": source_type,
                "metadata": metadata or {},
            }
            for chunk, h in fresh
        ]
        with self._lock:
            self._store.upsert_chunks(records, vectors)
            self._invalidate()
        return len(records)

    def delete_source(self, source: str) -> int:
        with self._lock:
            removed = self._store.delete_source(source)
            if removed:
                self._invalidate()
            return removed

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            reset_kb_version()
            self._invalidate()
            logger.info("Knowledge base cleared.")

    # ─── Retrieval ──────────────────────────────────────────────────────────
    def hybrid_search(self, query: str, source_filter: str | None = None, k: int | None = None) -> list[dict]:
        """Return up to ``k`` chunk dicts (with ``score``) most relevant to ``query``."""
        settings = get_settings()
        k = k or settings.top_k_final
        with self._lock:
            pool = self.list_chunks(source_filter)
            if not pool:
                return []
            bm25 = self._bm25_for(pool, source_filter)

        q_vec = gemini_embed_query(query)
        with self._lock:
            sem_hits = self._store.semantic_search(q_vec, source_filter, settings.top_k_coarse)
        semantic = {h["id"]: max(0.0, float(h["score"])) for h in sem_hits}

        raw = bm25.get_scores(tokenize(query)) if tokenize(query) else np.zeros(len(pool))
        top = float(np.max(raw)) if len(raw) else 0.0
        keyword = {pool[i]["id"]: float(raw[i]) / top for i in range(len(pool)) if top > 0 and raw[i] > 0}

        ranked = fuse_scores(semantic, keyword, settings.semantic_weight, settings.bm25_weight)
        by_id = {c["id"]: c for c in pool}
        by_id.update({h["id"]: h for h in sem_hits if h["id"] not in by_id})
        candidates = [
            {**{key: v for key, v in by_id[cid].items() if key != "score"}, "score": round(score, 4)}
            for cid, score in ranked[: settings.top_k_coarse]
            if cid in by_id
        ]

        if settings.rerank and len(candidates) > k:
            try:
                order = parse_rerank(
                    generate_content(rerank_prompt(query, [c["text"] for c in candidates], k)),
                    len(candidates),
                    k,
                )
                if order:
                    reranked = [candidates[i] for i in order]
                    # Top up with fused order if the model returned fewer than k.
                    reranked += [c for c in candidates if c not in reranked][: k - len(reranked)]
                    candidates = reranked
            except Exception as exc:  # re-ranking is best effort
                logger.warning("Re-rank failed, using fused ranking: %s", exc)
        return candidates[:k]

    def get_embeddings_for_clustering(self) -> tuple[list[dict], np.ndarray]:
        with self._lock:
            pairs = self._store.scroll_with_vectors()
        if not pairs:
            return [], np.empty((0, 0), dtype="float32")
        chunks, vectors = zip(*pairs)
        return list(chunks), np.asarray(vectors, dtype="float32")


_kb_singleton: KnowledgeBase | None = None
_kb_lock = threading.Lock()


def get_knowledge_base() -> KnowledgeBase:
    global _kb_singleton
    with _kb_lock:
        if _kb_singleton is None:
            _kb_singleton = KnowledgeBase()
        return _kb_singleton


def reset_knowledge_base() -> None:
    global _kb_singleton
    with _kb_lock:
        _kb_singleton = None
