import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# Mock Gemini before importing app
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-12345")


@pytest.fixture
def client():
    # Patch Gemini configure + embedding so app loads without real API key
    with patch("google.generativeai.configure"), \
         patch("google.generativeai.embed_content", return_value={"embedding": [[0.0]*768]}):
        import importlib, app as app_module
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

    def test_get_sources_empty(self, client):
        c, app_mod = client
        app_mod.kb.clear(save=False)
        r = c.get("/api/get-sources")
        data = json.loads(r.data)
        assert data["success"] is True
        assert data["sources"] == []

    def test_clear_data(self, client):
        c, _ = client
        r = c.post("/api/clear-data")
        data = json.loads(r.data)
        assert data["success"] is True


class TestContentIngestion:
    def test_add_text_content(self, client):
        c, _ = client
        mock_vec = np.array([[0.0]*768], dtype=np.float32)
        with patch("app.gemini_embed", return_value=mock_vec), \
             patch("google.generativeai.embed_content", return_value={"embedding": [[0.0]*768]}):
            r = c.post("/api/add-content",
                       data=json.dumps({"source": "Hello world test content " * 20,
                                        "source_type": "text"}),
                       content_type="application/json")
            data = json.loads(r.data)
            assert data["success"] is True

    def test_add_missing_source_type(self, client):
        c, _ = client
        r = c.post("/api/add-content",
                   data=json.dumps({"source": "test", "source_type": "unknown"}),
                   content_type="application/json")
        data = json.loads(r.data)
        assert data["success"] is False


class TestTools:
    def test_ask_question_empty_kb(self, client):
        c, app_mod = client
        app_mod.kb.clear(save=False)
        with patch("app.gemini_embed_query", return_value=[[0.0]*768]):
            r = c.post("/api/ask-question",
                       data=json.dumps({"question": "What is this?", "source": None}),
                       content_type="application/json")
            data = json.loads(r.data)
            assert data["success"] is True
            assert "information" in data["answer"].lower()

    def test_generate_invalid_tool(self, client):
        c, _ = client
        r = c.post("/api/generate-tool-content",
                   data=json.dumps({"tool": "invalid_tool", "topic": "test", "source": None}),
                   content_type="application/json")
        data = json.loads(r.data)
        assert data["success"] is False

    def test_generate_tool_no_topic(self, client):
        c, _ = client
        r = c.post("/api/generate-tool-content",
                   data=json.dumps({"tool": "flashcards", "topic": "", "source": None}),
                   content_type="application/json")
        data = json.loads(r.data)
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
        r = c.post("/api/summarize",
                   data=json.dumps({"source": None}),
                   content_type="application/json")
        data = json.loads(r.data)
        assert data["success"] is False

    def test_chat_clear(self, client):
        c, _ = client
        r = c.post("/api/chat/clear")
        data = json.loads(r.data)
        assert data["success"] is True
