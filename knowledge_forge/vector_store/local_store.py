"""Local FAISS + JSON persistence for single-node deployments."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any

import faiss
import numpy as np

from knowledge_forge.config import get_settings
from knowledge_forge.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class LocalFaissStore(VectorStore):
    """Chunks are stored in a JSON file; vectors in a FAISS inner-product index.

    Vectors are L2-normalised so inner product == cosine similarity. Row ``i`` of the
    index always corresponds to ``self.chunks[i]``.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.dim = settings.embed_dim
        self.kb_file = settings.kb_file
        self.index_file = settings.index_file
        self.ivf_threshold = settings.faiss_ivf_threshold
        self.ivf_nlist = settings.faiss_ivf_nlist
        self.nprobe = settings.faiss_nprobe
        self._lock = threading.RLock()
        self.chunks: list[dict] = []
        self.index: faiss.Index = self._new_flat_index()
        self._use_ivf = False
        self.load()

    # ─── Index helpers ──────────────────────────────────────────────────────
    def _new_flat_index(self) -> faiss.Index:
        return faiss.IndexFlatIP(self.dim)

    def _all_vectors(self) -> np.ndarray:
        if self.index.ntotal == 0:
            return np.empty((0, self.dim), dtype="float32")
        if self._use_ivf:
            self.index.make_direct_map()
        return np.vstack([self.index.reconstruct(i) for i in range(self.index.ntotal)]).astype("float32")

    def _rebuild_index(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.size == 0:
            self.index = self._new_flat_index()
            self._use_ivf = False
            return
        faiss.normalize_L2(vectors)
        if vectors.shape[0] >= self.ivf_threshold:
            nlist = min(self.ivf_nlist, max(1, vectors.shape[0] // 10))
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.add(vectors)
            index.nprobe = min(self.nprobe, nlist)
            index.make_direct_map()
            self.index = index
            self._use_ivf = True
            logger.info("Built IVF index (nlist=%d) for %d vectors", nlist, vectors.shape[0])
        else:
            self.index = self._new_flat_index()
            self.index.add(vectors)
            self._use_ivf = False

    # ─── Persistence ────────────────────────────────────────────────────────
    def load(self) -> None:
        has_kb, has_index = os.path.exists(self.kb_file), os.path.exists(self.index_file)
        if not (has_kb or has_index):
            return
        if has_kb != has_index:
            self._quarantine("only one of the two knowledge-base files exists")
            return
        try:
            with open(self.kb_file, encoding="utf-8") as f:
                chunks = json.load(f)
            index = faiss.read_index(self.index_file)
            if index.d != self.dim:
                raise ValueError(f"index dimension {index.d} != EMBED_DIM {self.dim}")
            if index.ntotal != len(chunks):
                raise ValueError(f"index has {index.ntotal} vectors but {len(chunks)} chunks")
            self.chunks = chunks
            self.index = index
            self._use_ivf = isinstance(index, faiss.IndexIVFFlat)
            if self._use_ivf:
                self.index.make_direct_map()
                self.index.nprobe = min(self.nprobe, self.index.nlist)
            logger.info("Loaded %d local chunks from %s.", len(self.chunks), self.kb_file)
        except Exception as exc:
            self._quarantine(str(exc))

    def _quarantine(self, reason: str) -> None:
        """Move unreadable files aside (instead of overwriting them on the next save)."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for path in (self.kb_file, self.index_file):
            if os.path.exists(path):
                backup = f"{path}.corrupt-{stamp}"
                os.replace(path, backup)
                logger.error("Local knowledge base unreadable (%s); moved %s to %s", reason, path, backup)
        self.chunks = []
        self.index = self._new_flat_index()
        self._use_ivf = False

    @staticmethod
    def _atomic_write(path: str, writer) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        os.close(fd)
        try:
            writer(tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def save(self) -> None:
        if not self.chunks:
            self._remove_files()
            return

        def write_kb(path: str) -> None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)

        self._atomic_write(self.kb_file, write_kb)
        self._atomic_write(self.index_file, lambda path: faiss.write_index(self.index, path))

    def _remove_files(self) -> None:
        for path in (self.kb_file, self.index_file):
            if os.path.exists(path):
                os.remove(path)

    # ─── VectorStore API ────────────────────────────────────────────────────
    def upsert_chunks(self, records: list[dict], vectors: np.ndarray) -> None:
        if not records:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32").reshape(len(records), self.dim)
        faiss.normalize_L2(vectors)
        new_total = len(self.chunks) + len(records)
        needs_rebuild = self._use_ivf or new_total >= self.ivf_threshold
        if needs_rebuild:
            all_vecs = np.vstack([self._all_vectors(), vectors])
            self.chunks.extend(records)
            self._rebuild_index(all_vecs)
        else:
            self.chunks.extend(records)
            self.index.add(vectors)
        self.save()

    def semantic_search(self, query_vector: np.ndarray, source_filter: str | None, k: int) -> list[dict[str, Any]]:
        if self.index.ntotal == 0 or k <= 0:
            return []
        q_vec = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q_vec)

        if not source_filter:
            distances, ids = self.index.search(q_vec, min(k, self.index.ntotal))
            return [
                {**self.chunks[int(idx)], "score": float(distances[0][j])}
                for j, idx in enumerate(ids[0])
                if idx >= 0
            ]

        pool = [i for i, c in enumerate(self.chunks) if c.get("source") == source_filter]
        if not pool:
            return []
        all_vecs = self._all_vectors()
        sub = all_vecs[pool]
        scores = sub @ q_vec[0]
        order = np.argsort(-scores)[:k]
        return [{**self.chunks[pool[int(i)]], "score": float(scores[int(i)])} for i in order]

    def scroll_chunks(self, source_filter: str | None = None) -> list[dict]:
        if source_filter:
            return [c for c in self.chunks if c.get("source") == source_filter]
        return list(self.chunks)

    def scroll_with_vectors(self) -> list[tuple[dict, np.ndarray]]:
        vecs = self._all_vectors()
        return list(zip(self.chunks, vecs))

    def delete_source(self, source: str) -> int:
        if not source:
            return 0
        return self._delete_where(lambda c: c.get("source") == source)

    def delete_ids(self, ids: list[str]) -> int:
        wanted = set(ids)
        if not wanted:
            return 0
        return self._delete_where(lambda c: c.get("id") in wanted)

    def _delete_where(self, predicate) -> int:
        keep = [i for i, c in enumerate(self.chunks) if not predicate(c)]
        removed = len(self.chunks) - len(keep)
        if not removed:
            return 0
        vecs = self._all_vectors()[keep] if keep else np.empty((0, self.dim), dtype="float32")
        self.chunks = [self.chunks[i] for i in keep]
        self._rebuild_index(vecs)
        self.save()
        return removed

    def count(self) -> int:
        return int(self.index.ntotal)

    def clear(self) -> None:
        self.chunks = []
        self.index = self._new_flat_index()
        self._use_ivf = False
        self._remove_files()

    def health_check(self) -> bool:
        return True

    def stats(self) -> dict:
        return {
            "backend": "local",
            "chunks": len(self.chunks),
            "vectors": int(self.index.ntotal),
            "sources": len({c.get("source") for c in self.chunks}),
            "index_type": "ivf" if self._use_ivf else "flat",
        }
