import re

import numpy as np
import pytest

from knowledge_forge.vector_store.local_store import LocalFaissStore
from tests.conftest import DIM, fake_vectors


def _records(texts, source="s1"):
    return [{"id": f"{source}-{i}", "text": t, "hash": t, "source": source, "source_type": "text",
             "metadata": {}} for i, t in enumerate(texts)]


def test_local_store_roundtrip_and_persistence():
    store = LocalFaissStore()
    texts = ["cats purr softly", "dogs bark loudly", "fish swim quietly"]
    store.upsert_chunks(_records(texts), fake_vectors(texts))
    hits = store.semantic_search(fake_vectors(["dogs bark"])[0], None, 2)
    assert hits[0]["text"] == "dogs bark loudly"
    reloaded = LocalFaissStore()
    assert reloaded.count() == 3


def test_local_store_filter_delete_and_clear():
    store = LocalFaissStore()
    store.upsert_chunks(_records(["alpha one", "alpha two"], "a"), fake_vectors(["alpha one", "alpha two"]))
    store.upsert_chunks(_records(["beta one"], "b"), fake_vectors(["beta one"]))
    hits = store.semantic_search(fake_vectors(["alpha one"])[0], "b", 5)
    assert [h["source"] for h in hits] == ["b"]
    assert store.delete_source("a") == 2
    assert store.count() == 1
    assert store.semantic_search(fake_vectors(["beta one"])[0], None, 5)[0]["text"] == "beta one"
    store.clear()
    assert store.count() == 0
    assert LocalFaissStore().count() == 0


def test_local_store_rejects_dimension_mismatch(tmp_path, monkeypatch):
    store = LocalFaissStore()
    store.upsert_chunks(_records(["x y"]), fake_vectors(["x y"]))
    from knowledge_forge import config

    monkeypatch.setenv("EMBED_DIM", str(DIM * 2))
    config.get_settings.cache_clear()
    assert LocalFaissStore().count() == 0  # incompatible index is ignored, not crashed on


def test_kb_add_dedupes_and_skips_reembedding(kb, fake_embed):
    chunks = ["photosynthesis makes sugar", "the calvin cycle fixes carbon"]
    assert kb.add_chunks(chunks, "bio", "text") == 2
    assert kb.add_chunks(chunks, "bio", "text") == 0
    assert len(fake_embed) == 1  # second add did not call the embedder
    assert kb.add_chunks(chunks + ["new fact about leaves"], "bio", "text") == 1
    assert fake_embed[-1] == ["new fact about leaves"]
    assert kb.list_sources() == [{"source": "bio", "type": "text", "chunks": 3}]


def test_kb_hybrid_search_returns_scored_hits(kb):
    kb.add_chunks(["photosynthesis makes sugar in leaves", "volcanoes erupt lava"], "geo-bio", "text")
    kb.add_chunks(["the calvin cycle fixes carbon dioxide"], "notes", "text")
    hits = kb.hybrid_search("photosynthesis sugar", k=2)
    assert hits[0]["text"] == "photosynthesis makes sugar in leaves"
    assert {"id", "text", "source", "source_type", "score"} <= set(hits[0])
    scoped = kb.hybrid_search("photosynthesis sugar", source_filter="notes", k=2)
    assert {h["source"] for h in scoped} == {"notes"}


def test_kb_rerank_reorders_and_survives_failure(kb, monkeypatch):
    from knowledge_forge import config, knowledge_base

    monkeypatch.setenv("RERANK", "true")
    monkeypatch.setenv("TOP_K_FINAL", "1")
    config.get_settings.cache_clear()
    kb.add_chunks(["apple pie recipe", "apple tree growth", "apple stock price"], "s", "text")

    def pick_stock(prompt):
        # Candidate order depends on score ties, so find the passage number by its text.
        n = re.search(r"\[(\d+)\] apple stock price", prompt).group(1)
        return f"[{n}]"

    monkeypatch.setattr(knowledge_base, "generate_content", pick_stock)
    assert kb.hybrid_search("apple")[0]["text"] == "apple stock price"

    def boom(_p):
        raise RuntimeError("model down")

    monkeypatch.setattr(knowledge_base, "generate_content", boom)
    assert len(kb.hybrid_search("apple")) == 1


def test_kb_delete_and_clear(kb):
    kb.add_chunks(["one fish", "two fish"], "a", "text")
    kb.add_chunks(["red fish"], "b", "text")
    assert kb.delete_source("a") == 2
    assert [s["source"] for s in kb.list_sources()] == ["b"]
    assert kb.delete_source("missing") == 0
    kb.clear()
    assert kb.count() == 0
    assert kb.hybrid_search("fish") == []


def test_kb_clustering_vectors(kb):
    kb.add_chunks([f"topic {i} words" for i in range(6)], "s", "text")
    chunks, vecs = kb.get_embeddings_for_clustering()
    assert len(chunks) == 6 and vecs.shape == (6, DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_fuse_scores_weights():
    from knowledge_forge.knowledge_base import fuse_scores

    ranked = fuse_scores({"a": 1.0, "b": 0.2}, {"b": 1.0}, 0.7, 0.3)
    assert ranked[0][0] == "a"
    assert ranked[1] == ("b", pytest.approx(0.44))
