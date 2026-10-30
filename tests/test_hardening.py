"""Regression tests for the second-pass backend fixes."""
import json
import logging
import os

import pytest

from knowledge_forge import cache, config, llm, web
from knowledge_forge.errors import RateLimitError
from knowledge_forge.knowledge_base import KnowledgeBase
from knowledge_forge.logging_setup import RedactingFilter
from knowledge_forge.vector_store.local_store import LocalFaissStore
from tests.conftest import fake_vectors


@pytest.fixture
def hkb(fake_embed):
    return KnowledgeBase()


@pytest.fixture
def hclient(hkb, monkeypatch):
    monkeypatch.setattr(web, "generate_content", lambda prompt, **_k: "Light becomes sugar [1].")
    app = web.create_app(kb=hkb)
    app.config["TESTING"] = True
    return app.test_client()


def _seed(kb):
    kb.add_chunks(["photosynthesis converts light into sugar", "chlorophyll absorbs light"], "bio", "text")


# ─── Knowledge base ─────────────────────────────────────────────────────────
def test_sync_source_replaces_stale_chunks(hkb):
    assert hkb.sync_source(["old fact one", "old fact two"], "page", "web") == (2, 0)
    added, removed = hkb.sync_source(["old fact one", "brand new fact"], "page", "web")
    assert (added, removed) == (1, 1)
    texts = {h["text"] for h in hkb.hybrid_search("fact", k=10)}
    assert texts == {"old fact one", "brand new fact"}


def test_sync_source_dedupes_within_batch(hkb, fake_embed):
    hkb.sync_source(["same text", "same text"], "s", "text")
    assert hkb.count() == 1 and fake_embed[-1] == ["same text"]


def test_clear_only_bumps_version(hkb):
    _seed(hkb)
    before = cache.get_kb_version()
    hkb.clear()
    assert cache.get_kb_version() > before


def test_delete_empty_source_is_noop(hkb):
    _seed(hkb)
    assert hkb.delete_source("") == 0 and hkb.count() == 2


# ─── Local store ────────────────────────────────────────────────────────────
def test_corrupt_local_store_is_quarantined_not_overwritten():
    store = LocalFaissStore()
    store.upsert_chunks([{"id": "a", "text": "t", "hash": "h", "source": "s", "source_type": "text",
                          "metadata": {}}], fake_vectors(["t"]))
    with open(store.kb_file, "w") as f:
        f.write("{not json")
    fresh = LocalFaissStore()
    assert fresh.count() == 0
    leftovers = os.listdir(os.path.dirname(store.kb_file))
    assert any(".corrupt-" in name for name in leftovers)


def test_local_store_delete_ids():
    store = LocalFaissStore()
    recs = [{"id": f"i{i}", "text": t, "hash": t, "source": "s", "source_type": "text", "metadata": {}}
            for i, t in enumerate(["aa", "bb"])]
    store.upsert_chunks(recs, fake_vectors(["aa", "bb"]))
    store.delete_ids(["i0"])
    assert store.count() == 1


# ─── Web ────────────────────────────────────────────────────────────────────
def test_unknown_source_scope_is_404(hclient, hkb):
    _seed(hkb)
    r = hclient.post("/api/ask-question", json={"question": "light", "source": "ghost"})
    assert r.status_code == 404


def test_citations_mark_cited_and_grounded(hclient, hkb):
    _seed(hkb)
    data = hclient.post("/api/ask-question", json={"question": "light"}).get_json()
    assert data["sources"][0]["cited"] is True
    assert data["grounded"] is True


def test_rate_limiter_returns_429(monkeypatch, fake_embed):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    config.get_settings.cache_clear()
    monkeypatch.setattr(web, "generate_content", lambda prompt, **_k: "ok")
    kb = KnowledgeBase()
    _seed(kb)
    c = web.create_app(kb=kb).test_client()
    codes = [c.post("/api/ask-question", json={"question": "light"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limiter_unit():
    lim = web.RateLimiter(1)
    lim.check("a")
    lim.check("b")
    with pytest.raises(RateLimitError):
        lim.check("a")
    web.RateLimiter(0).check("x")  # disabled


def _events(body: str):
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_chat_stream_emits_meta_tokens_done(hclient, hkb, monkeypatch):
    _seed(hkb)
    monkeypatch.setattr(web, "stream_content", lambda prompt, **_k: iter(["Sugar ", "[1]."]))
    r = hclient.post("/api/chat/stream", json={"message": "light?"})
    assert r.status_code == 200 and r.mimetype == "text/event-stream"
    events = _events(r.get_data(as_text=True))
    assert [e for e, _ in events] == ["meta", "token", "token", "done"]
    assert events[-1][1]["answer"] == "Sugar [1]." and events[-1][1]["grounded"]


def test_chat_stream_reports_midstream_error(hclient, hkb, monkeypatch):
    from knowledge_forge.errors import LLMError

    _seed(hkb)

    def broken(prompt, **_k):
        yield "partial"
        raise LLMError("The AI service is unavailable.")

    monkeypatch.setattr(web, "stream_content", broken)
    events = _events(hclient.post("/api/chat/stream", json={"message": "x"}).get_data(as_text=True))
    assert events[-1][0] == "error" and events[-1][1]["status"] == 502


def test_quiz_tool_returns_structured_questions(hclient, hkb, monkeypatch):
    _seed(hkb)
    raw = json.dumps({"questions": [{"question": "What makes sugar?", "options": ["Roots", "Light", "Rocks"],
                                     "answer": "B", "explanation": "Photosynthesis [1]."}]})
    monkeypatch.setattr(web, "generate_content", lambda prompt, **_k: raw)
    data = hclient.post("/api/generate-tool-content", json={"tool": "quiz", "topic": "light"}).get_json()
    assert data["content"][0]["answer"] == 1 and len(data["content"][0]["options"]) == 3
    alias = hclient.post("/api/generate-tool-content", json={"tool": "mock_test", "topic": "light"})
    assert alias.status_code == 200


def test_structured_tool_repairs_bad_json_once(hclient, hkb, monkeypatch):
    _seed(hkb)
    replies = iter(["not json at all", '[{"question": "Q", "answer": "A"}]'])
    monkeypatch.setattr(web, "generate_content", lambda prompt, **_k: next(replies))
    data = hclient.post("/api/generate-tool-content", json={"tool": "flashcards", "topic": "x"}).get_json()
    assert data["content"] == [{"question": "Q", "answer": "A"}]


def test_upload_rejects_disguised_binary(hclient):
    import io

    r = hclient.post("/api/upload-file", data={"file": (io.BytesIO(b"MZ\x90\x00binary"), "notes.pdf")},
                     content_type="multipart/form-data")
    assert r.status_code in (400, 415, 422)


def test_api_responses_are_not_cached(hclient):
    r = hclient.get("/api/sources")
    assert r.headers["Cache-Control"] == "no-store"
    assert "default-src" in hclient.get("/").headers.get("Content-Security-Policy", "")
    assert r.headers.get("X-Request-ID")


# ─── LLM helpers ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("answer,expected", [("C", 2), (2, 2), ("Light", 1), ("1", 1)])
def test_validate_quiz_answer_forms(answer, expected):
    q = llm.validate_quiz([{"question": "q", "options": ["Roots", "Light", "Rocks"], "answer": answer}])
    assert q[0]["answer"] == expected


def test_validate_quiz_drops_invalid_items():
    assert llm.validate_quiz([{"question": "q", "options": ["one"], "answer": 0}, "junk"]) == []


def test_parse_json_output_is_lenient():
    assert llm.parse_json_output('Sure! ```json\n{"a": [1, 2,],}\n``` done') == {"a": [1, 2]}


def test_normalize_mermaid_strips_directives_and_adds_header():
    out = llm.normalize_mermaid("A[Start] --> B[End]\nclick A call evil()\nstyle A fill:#f00\n%%{init: {}}%%",
                                "flowchart")
    assert out.startswith("graph TD") and "click" not in out and "style" not in out and "%%" not in out
    assert llm.normalize_mermaid("root((Topic))", "mindmap").startswith("mindmap")


# ─── Logging / settings ─────────────────────────────────────────────────────
def test_redacting_filter_hides_secrets():
    f = RedactingFilter(["super-secret-value"])
    fake_google_key = "AIza" + "Fake" * 9  # built at runtime so scanners don't flag a literal
    rec = logging.LogRecord("x", logging.INFO, __file__, 1,
                            f"key=abc123 {fake_google_key} super-secret-value %s",
                            ("password=hunter2",), None)
    f.filter(rec)
    msg = rec.getMessage()
    for leaked in ("abc123", fake_google_key, "super-secret-value", "hunter2"):
        assert leaked not in msg


def test_new_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("CRAWL_MAX_REDIRECTS", "2")
    monkeypatch.setenv("YOUTUBE_LANGUAGES", "de, en")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")
    monkeypatch.setenv("FLASK_DEBUG", "true")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.llm_temperature == 0.7 and s.crawl_max_redirects == 2
    assert list(s.youtube_languages) == ["de", "en"] and s.rate_limit_per_minute == 5 and s.debug is True


def test_invalid_numeric_env_falls_back(monkeypatch):
    monkeypatch.setenv("CRAWL_MAX_BYTES", "lots")
    config.get_settings.cache_clear()
    assert config.get_settings().crawl_max_bytes == 5 * 1024 * 1024
