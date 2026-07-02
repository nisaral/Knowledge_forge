"""Local FAISS + JSON persistence for single-node development."""
import json
import logging
import os
import tempfile
from typing import Any

import faiss
import numpy as np

from config import EMBED_DIM, FAISS_IVF_NLIST, FAISS_IVF_THRESHOLD, INDEX_FILE, KB_FILE
from services.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


class LocalFaissStore(VectorStore):
    def __init__(self):
        self.dim = EMBED_DIM
        self.chunks: list[dict] = []
        self.index = faiss.IndexFlatIP(self.dim)
        self._use_ivf = False
        self._id_to_idx: dict[str, int] = {}
        self.load()

    def _new_flat_index(self) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(self.dim)

    def _build_ivf_index(self, vectors: np.ndarray) -> faiss.Index:
        nlist = min(FAISS_IVF_NLIST, max(1, vectors.shape[0] // 10))
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = min(16, nlist)
        return index

    def _rebuild_index(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            self.index = self._new_flat_index()
            self._use_ivf = False
            return
        faiss.normalize_L2(vectors)
        if len(self.chunks) >= FAISS_IVF_THRESHOLD:
            logger.info("Building IVF index for %d chunks", len(self.chunks))
            self.index = self._build_ivf_index(vectors)
            self._use_ivf = True
        else:
            self.index = self._new_flat_index()
            self.index.add(vectors)
            self._use_ivf = False

    def _reindex_id_map(self) -> None:
        self._id_to_idx = {c["id"]: i for i, c in enumerate(self.chunks)}

    def load(self) -> None:
        try:
            if os.path.exists(KB_FILE) and os.path.exists(INDEX_FILE):
                with open(KB_FILE, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                self.index = faiss.read_index(INDEX_FILE)
                self._use_ivf = isinstance(self.index, faiss.IndexIVFFlat)
                self._reindex_id_map()
                logger.info("Loaded %d local chunks from disk.", len(self.chunks))
        except Exception as exc:
            logger.warning("Could not load local KB (%s). Starting fresh.", exc)
            self.clear()

    def _atomic_write(self, path: str, writer) -> None:
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
        try:
            def write_kb(path):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            self._atomic_write(KB_FILE, write_kb)
            if self.index.ntotal > 0:
                def write_index(path):
                    faiss.write_index(self.index, path)
                self._atomic_write(INDEX_FILE, write_index)
        except Exception as exc:
            logger.error("Error saving local KB: %s", exc)

    def upsert_chunks(self, records: list[dict], vectors: np.ndarray) -> None:
        faiss.normalize_L2(vectors)
        start_len = len(self.chunks)
        self.chunks.extend(records)
        self._reindex_id_map()

        if (
            self.index.ntotal == 0
            or (len(self.chunks) >= FAISS_IVF_THRESHOLD and not self._use_ivf)
            or (start_len < FAISS_IVF_THRESHOLD <= len(self.chunks))
        ):
            if self.index.ntotal:
                old_vecs = np.array(
                    [self.index.reconstruct(i) for i in range(self.index.ntotal)],
                    dtype="float32",
                )
                all_vecs = np.vstack([old_vecs, vectors])
            else:
                all_vecs = vectors
            self._rebuild_index(all_vecs)
        else:
            self.index.add(vectors)
        self.save()

    def semantic_search(
        self,
        query_vector: np.ndarray,
        source_filter: str | None,
        k: int,
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        pool_indices = (
            [i for i, c in enumerate(self.chunks) if c["source"] == source_filter]
            if source_filter
            else list(range(len(self.chunks)))
        )
        if not pool_indices:
            return []

        q_vec = query_vector.astype("float32")
        faiss.normalize_L2(q_vec)

        if len(pool_indices) == self.index.ntotal:
            distances, result_ids = self.index.search(q_vec, min(k, self.index.ntotal))
            hits = []
            for j, idx in enumerate(result_ids[0]):
                if idx < 0:
                    continue
                chunk = self.chunks[int(idx)]
                hits.append({**chunk, "score": float(distances[0][j])})
            return hits

        sub_vecs = np.array(
            [self.index.reconstruct(i) for i in pool_indices],
            dtype="float32",
        )
        sub_idx = faiss.IndexFlatIP(self.dim)
        sub_idx.add(sub_vecs)
        sem_k = min(k, len(pool_indices))
        distances, result_ids = sub_idx.search(q_vec, sem_k)
        hits = []
        for j, local_i in enumerate(result_ids[0]):
            if local_i < 0:
                continue
            chunk = self.chunks[pool_indices[int(local_i)]]
            hits.append({**chunk, "score": float(distances[0][j])})
        return hits

    def scroll_chunks(self, source_filter: str | None = None) -> list[dict]:
        if source_filter:
            return [c for c in self.chunks if c["source"] == source_filter]
        return list(self.chunks)

    def scroll_with_vectors(self) -> list[tuple[dict, np.ndarray]]:
        if self.index.ntotal == 0:
            return []
        pairs = []
        for i in range(self.index.ntotal):
            vec = self.index.reconstruct(i)
            pairs.append((self.chunks[i], np.array(vec, dtype="float32")))
        return pairs

    def count(self) -> int:
        return self.index.ntotal

    def clear(self) -> None:
        self.chunks = []
        self.index = self._new_flat_index()
        self._use_ivf = False
        self._id_to_idx = {}
        for path in (KB_FILE, INDEX_FILE):
            if os.path.exists(path):
                os.remove(path)

    def health_check(self) -> bool:
        return True

    def stats(self) -> dict:
        sources = {c["source"] for c in self.chunks}
        return {
            "backend": "local",
            "chunks": len(self.chunks),
            "vectors": self.index.ntotal,
            "sources": len(sources),
            "index_type": "ivf" if self._use_ivf else "flat",
        }