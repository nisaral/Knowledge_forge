import io

import pytest

from knowledge_forge import web
from knowledge_forge.knowledge_base import KnowledgeBase


@pytest.fixture
def app_kb(fake_embed):
    return KnowledgeBase()


@pytest.fixture
def client(app_kb, monkeypatch):
    monkeypatch.setattr(web, "generate_content", lambda prompt, **_k: "Answer citing [1].")
    app = web.create_app(kb=app_kb)
    app.config["TESTING"] = True
    return app.test_client()


def seed(kb):
    kb.add_chunks(["photosynthesis converts light into sugar", "chlorophyll absorbs light",
                   "the calvin cycle fixes carbon", "plants release oxygen", "roots absorb water"],
                  "bio-notes", "text")


def test_health_and_index(client):
    assert client.get("/api/health").get_json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200 and b"KnowledgeForge" in page.data
    assert page.headers["X-Content-Type-Options"] == "nosniff"


def test_ready_reports_components(client):
    data = client.get("/api/ready").get_json()
    assert data["gemini_configured"] is True and data["vector_store_healthy"] is True


def test_missing_key_gives_clear_503(client, monkeypatch):
    from knowledge_forge import config

    monkeypatch.delenv("GEMINI_API_KEY")
    config.get_settings.cache_clear()
    r = client.post("/api/ask-question", json={"question": "hi"})
    assert r.status_code == 503 and "GEMINI_API_KEY" in r.get_json()["message"]
    assert client.get("/api/get-sources").get_json()["gemini_configured"] is False


@pytest.mark.parametrize(
    "path, body",
    [
        ("/api/ask-question", {}),
        ("/api/ask-question", {"question": 5}),
        ("/api/chat", {"message": "   "}),
        ("/api/add-content", {"source": "x", "source_type": "ftp"}),
        ("/api/generate-tool-content", {"tool": "bogus", "topic": "t"}),
        ("/api/delete-source", {}),
    ],
)
def test_validation_errors(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 400 and r.get_json()["success"] is False


def test_non_json_body_rejected(client):
    r = client.post("/api/ask-question", data="nope", content_type="text/plain")
    assert r.status_code == 400


def test_question_too_long(client):
    r = client.post("/api/ask-question", json={"question": "x" * 5000})
    assert r.status_code == 400


def test_add_text_then_list_and_delete(client):
    r = client.post("/api/add-content", json={"source": "Some study notes about cells. " * 5,
                                              "source_type": "text", "title": "Cell notes"})
    body = r.get_json()
    assert r.status_code == 200 and body["chunks_added"] >= 1 and body["source"] == "Cell notes"
    again = client.post("/api/add-content", json={"source": "Some study notes about cells. " * 5,
                                                  "source_type": "text", "title": "Cell notes"}).get_json()
    assert again["chunks_added"] == 0 and "already" in again["message"]
    sources = client.get("/api/get-sources").get_json()["sources"]
    assert sources[0]["source"] == "Cell notes"
    assert client.post("/api/delete-source", json={"source": "Cell notes"}).status_code == 200
    assert client.post("/api/delete-source", json={"source": "Cell notes"}).status_code == 404


def test_add_web_failure_is_422(client, monkeypatch):
    monkeypatch.setattr(web, "ingest_text_source", lambda s, t, *_a: (None, "blocked", "web"))
    r = client.post("/api/add-content", json={"source": "https://example.com", "source_type": "web"})
    assert r.status_code == 422 and r.get_json()["message"] == "blocked"


def test_upload_text_file(client):
    data = {"file": (io.BytesIO(b"Mitochondria are the powerhouse of the cell. " * 10), "notes.txt")}
    r = client.post("/api/upload-file", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["source_type"] == "text"


def test_upload_rejects_unknown_and_empty(client):
    bad = client.post("/api/upload-file", data={"file": (io.BytesIO(b"MZ"), "x.exe")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400
    empty = client.post("/api/upload-file", data={"file": (io.BytesIO(b""), "x.txt")},
                        content_type="multipart/form-data")
    assert empty.status_code == 400


def test_ask_returns_citations(client, app_kb):
    seed(app_kb)
    data = client.post("/api/ask-question", json={"question": "what does chlorophyll absorb"}).get_json()
    assert data["answer"] == "Answer citing [1]."
    assert data["sources"][0]["n"] == 1 and data["sources"][0]["source"] == "bio-notes"


def test_ask_on_empty_kb_is_graceful(client):
    data = client.post("/api/ask-question", json={"question": "anything"}).get_json()
    assert data["success"] and data["sources"] == []


def test_chat_uses_client_history(client, app_kb, monkeypatch):
    seed(app_kb)
    prompts = []
    monkeypatch.setattr(web, "generate_content", lambda p, **_k: prompts.append(p) or "ok")
    history = [{"role": "user", "content": "first question"}, {"role": "assistant", "content": "first answer"},
               {"role": "hacker", "content": 123}]
    r = client.post("/api/chat", json={"message": "follow up", "history": history})
    assert r.status_code == 200 and r.get_json()["answer"] == "ok"
    assert "first question" in prompts[0] and "first answer" in prompts[0]


def test_clean_history_limits_turns():
    hist = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    assert [h["content"] for h in web._clean_history(hist, 2)] == ["m16", "m17", "m18", "m19"]
    assert web._clean_history("nope", 3) == []


def test_tools(client, app_kb, monkeypatch):
    seed(app_kb)
    monkeypatch.setattr(web, "generate_content",
                        lambda p, **_k: '```json\n[{"question": "Q", "answer": "A"}]\n```' if "flashcards" in p
                        else "```mermaid\ngraph TD\nA[Light] --> B[Sugar]\n```")
    cards = client.post("/api/generate-tool-content", json={"tool": "flashcards", "topic": "light"}).get_json()
    assert cards["content"] == [{"question": "Q", "answer": "A"}]
    chart = client.post("/api/generate-tool-content", json={"tool": "flowchart", "topic": "light"}).get_json()
    assert chart["mermaid_code"].startswith("graph TD")


def test_tool_on_empty_kb(client):
    r = client.post("/api/generate-tool-content", json={"tool": "mock_test", "topic": "x"})
    assert r.status_code == 422


def test_summarize_and_entities(client, app_kb, monkeypatch):
    seed(app_kb)
    assert client.post("/api/summarize", json={}).get_json()["content"] == "Answer citing [1]."
    ents_json = '{"Concepts": ["Photosynthesis", " ", "Photosynthesis"]}'
    monkeypatch.setattr(web, "generate_content", lambda p, **_k: ents_json)
    ents = client.post("/api/extract-entities", json={}).get_json()["entities"]
    assert ents == {"Concepts": ["Photosynthesis"]}


def test_cluster_topics(client, app_kb, monkeypatch):
    seed(app_kb)
    monkeypatch.setattr(web, "generate_content", lambda p, **_k: "Plant biology")
    data = client.get("/api/cluster-topics").get_json()
    assert data["image"].startswith("data:image/png;base64,") and data["labels"]


def test_cluster_needs_content(client):
    assert client.get("/api/cluster-topics").status_code == 400


def test_llm_failure_maps_to_502(client, app_kb, monkeypatch):
    from knowledge_forge.errors import LLMError

    seed(app_kb)

    def fail(_p, **_k):
        raise LLMError("The AI service is unavailable.")

    monkeypatch.setattr(web, "generate_content", fail)
    r = client.post("/api/ask-question", json={"question": "light"})
    assert r.status_code == 502


def test_unexpected_error_is_500_without_details(client, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(web, "ingest_text_source", boom)
    r = client.post("/api/add-content", json={"source": "https://e.com", "source_type": "web"})
    assert r.status_code == 500 and "secret" not in r.get_json()["message"]


def test_clear_data(client, app_kb):
    seed(app_kb)
    assert client.post("/api/clear-data").status_code == 200
    assert app_kb.count() == 0


def test_cors_opt_in(monkeypatch, fake_embed):
    from knowledge_forge import config

    plain = web.create_app(kb=KnowledgeBase()).test_client()
    assert "Access-Control-Allow-Origin" not in plain.get("/api/health", headers={"Origin": "https://x.dev"}).headers
    monkeypatch.setenv("CORS_ORIGINS", "https://x.dev")
    config.get_settings.cache_clear()
    allowed = web.create_app(kb=KnowledgeBase()).test_client()
    assert allowed.get("/api/health", headers={"Origin": "https://x.dev"}).headers["Access-Control-Allow-Origin"] \
        == "https://x.dev"


def test_index_has_no_inline_scripts_for_csp(client):
    import re

    html = client.get("/").get_data(as_text=True)
    scripts = re.findall(r"<script\b([^>]*)>(.*?)</script>", html, flags=re.S)
    assert scripts, "expected external scripts"
    for attrs, body in scripts:
        assert "src=" in attrs and not body.strip(), "inline <script> would be blocked by script-src 'self'"
    assert "theme.js" in html
    assert 'data-tool="quiz"' in html and 'data-tool="mock_test"' not in html
