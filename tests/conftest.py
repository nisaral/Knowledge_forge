import hashlib

import numpy as np
import pytest

from knowledge_forge import cache, config, embeddings, gemini, graph, knowledge_base

DIM = 16


def fake_vectors(texts):
    """Deterministic pseudo-embeddings: bag of hashed words, so related texts are similar."""
    out = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().split():
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        if not out[row].any():
            out[row, 0] = 1.0
    return out


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets a clean data dir, a fake key and no Redis/Qdrant."""
    for name in ("REDIS_URL", "QDRANT_URL", "CORS_ORIGINS", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("FLASK_SECRET", "test-secret")
    monkeypatch.setenv("VECTOR_STORE", "local")
    monkeypatch.setenv("EMBED_DIM", str(DIM))
    monkeypatch.setenv("RERANK", "false")
    monkeypatch.setenv("CRAWL_RESPECT_ROBOTS", "false")
    monkeypatch.setattr(config, "load_dotenv", None, raising=False)
    config.get_settings.cache_clear()
    cache.reset_cache_client()
    cache.reset_kb_version()
    embeddings.reset_embedding_state()
    knowledge_base.reset_knowledge_base()
    gemini.reset_client()
    graph.reset_graph_cache()

    def _no_network_client():
        raise AssertionError("Test tried to reach Gemini; patch generate_content/gemini_embed instead.")

    monkeypatch.setattr(gemini, "get_client", _no_network_client)
    yield
    config.get_settings.cache_clear()
    knowledge_base.reset_knowledge_base()


@pytest.fixture
def fake_embed(monkeypatch):
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return fake_vectors(texts)

    monkeypatch.setattr(knowledge_base, "gemini_embed", embed)
    monkeypatch.setattr(knowledge_base, "gemini_embed_query", lambda q: fake_vectors([q])[0])
    return calls


@pytest.fixture
def kb(fake_embed):
    return knowledge_base.KnowledgeBase()
