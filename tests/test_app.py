import json
import os
import sys

import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-12345")
    monkeypatch.setenv("VECTOR_STORE", "local")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("SESSION_TYPE", "filesystem")
    monkeypatch.setenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "tmp_data"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    data_dir = str(tmp_path / "kb_data")
    monkeypatch.setenv("DATA_DIR", data_dir)

    with patch("google.generativeai.configure"), \
         patch("services.embeddings.genai.embed_content", return_value={"embedding": [[0.0] * 768]}):
        import importlib
        import config
        importlib.reload(config)
        import app as app_module
        importlib.reload(app_module)
        app_module.app.config["TESTING"] = True
        app_module.app.config["SECRET_KEY"] = "test"
        with app_module.app.test_client() as c:
            yield c, app_module


class TestHealthEndpoints:
    def test_index_returns_200(self, client):
        c, _ = client
        r = c.get("/")
        assert r.status_code == 200

    def test_health_endpoint(self, client):
        c, _ = client
        r = c.get("/api/health")
        data = json.loads(r.data)
        assert r.status_code == 200
        assert data["status"] == "ok"

    def test_ready_endpoint(self, client):
        c, _ = client
        r = c.get("/api/ready")
        data = json.loads(r.data)
        assert data["gemini_configured"] is True
        assert data["vector_store"] == "local"
        assert data["vector_store_healthy"] is True
        assert data["redis_healthy"] is True
        assert "knowledge_base" in data

    def test_get_sources_empty(self, client):
        c, app_mod = client
        app_mod.kb.clear(save=False)
        r = c.get("/api/get-sources")
        data = json.loads(r.data)
        assert data["success"] is True
        assert data["sources"] == []
        assert data["stats"]["chunks"] == 0
        assert data["stats"]["backend"] == "local"

    def test_clear_data(self, client):
        c, _ = client
        r = c.post("/api/clear-data")
        data = json.loads(r.data)
        assert data["success"] is True


class TestContentIngestion:
    def test_add_text_content(self, client):
        c, _ = client
        mock_vec = np.array([[0.0] * 768], dtype=np.float32)
        with patch("services.knowledge_base.gemini_embed", return_value=mock_vec):
            r = c.post(
                "/api/add-content",
                data=json.dumps({
                    "source": "Hello world test content " * 20,
                    "source_type": "text",
                }),
                content_type="application/json",
            )
            data = json.loads(r.data)
            assert r.status_code == 200
            assert data["success"] is True
            assert data["chunks_added"] > 0

    def test_add_missing_source(self, client):
        c, _ = client
        r = c.post(
            "/api/add-content",
            data=json.dumps({"source": "", "source_type": "text"}),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert r.status_code == 400
        assert data["success"] is False

    def test_add_missing_source_type(self, client):
        c, _ = client
        r = c.post(
            "/api/add-content",
            data=json.dumps({"source": "test", "source_type": "unknown"}),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert r.status_code == 400
        assert data["success"] is False


class TestTools:
    def test_ask_question_empty_kb(self, client):
        c, app_mod = client
        app_mod.kb.clear(save=False)
        with patch("services.knowledge_base.gemini_embed_query", return_value=np.array([[0.0] * 768], dtype=np.float32)):
            r = c.post(
                "/api/ask-question",
                data=json.dumps({"question": "What is this?", "source": None}),
                content_type="application/json",
            )
            data = json.loads(r.data)
            assert data["success"] is True
            assert "information" in data["answer"].lower()

    def test_generate_invalid_tool(self, client):
        c, _ = client
        r = c.post(
            "/api/generate-tool-content",
            data=json.dumps({"tool": "invalid_tool", "topic": "test", "source": None}),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert data["success"] is False

    def test_generate_tool_no_topic(self, client):
        c, _ = client
        r = c.post(
            "/api/generate-tool-content",
            data=json.dumps({"tool": "flashcards", "topic": "", "source": None}),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert r.status_code == 400
        assert data["success"] is False

    def test_cluster_topics_insufficient_data(self, client):
        c, app_mod = client
        app_mod.kb.clear(save=False)
        r = c.get("/api/cluster-topics")
        data = json.loads(r.data)
        assert data["success"] is False
        assert "5" in data["message"]

    def test_summarize_empty_kb(self, client):
        c, app_mod = client
        app_mod.kb.clear(save=False)
        r = c.post(
            "/api/summarize",
            data=json.dumps({"source": None}),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert r.status_code == 404
        assert data["success"] is False

    def test_chat_clear(self, client):
        c, _ = client
        r = c.post("/api/chat/clear")
        data = json.loads(r.data)
        assert data["success"] is True

    def test_chat_requires_message(self, client):
        c, _ = client
        r = c.post(
            "/api/chat",
            data=json.dumps({"message": "", "source": None}),
            content_type="application/json",
        )
        data = json.loads(r.data)
        assert r.status_code == 400
        assert data["success"] is False


class TestKnowledgeBase:
    def test_hybrid_search_empty(self, client):
        c, app_mod = client
        app_mod.kb.clear(save=False)
        assert app_mod.kb.hybrid_search("test query") == []

    def test_chunk_text_overlap(self):
        from services.ingestion import chunk_text
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(text, size=20, overlap=5)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 20 for c in chunks)