import pytest

from knowledge_forge import config
from knowledge_forge.chunking import chunk_text, content_hash, normalize_text


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert (s.chunk_size, s.chunk_overlap) == (100, 10)
    assert s.cors_origins == ("https://a.example", "https://b.example")
    assert s.gemini_configured


def test_invalid_values_fall_back(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    monkeypatch.setenv("CHUNK_OVERLAP", "9999")
    monkeypatch.setenv("VECTOR_STORE", "mystery")
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.chunk_size == 400
    assert s.chunk_overlap < s.chunk_size
    assert s.vector_store == "local"


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    config.get_settings.cache_clear()
    assert not config.get_settings().gemini_configured


def test_flask_secret_generated_when_unset(monkeypatch):
    monkeypatch.delenv("FLASK_SECRET")
    config.get_settings.cache_clear()
    secret = config.flask_secret(config.get_settings())
    assert len(secret) >= 32


def test_chunking_windows_and_overlap():
    words = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(words, size=10, overlap=2)
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w8"
    assert chunks[-1].split()[-1] == "w24"
    assert all(len(c.split()) <= 10 for c in chunks)


def test_chunking_edge_cases():
    assert chunk_text("", 10, 2) == []
    assert chunk_text("one two", 10, 2) == ["one two"]
    with pytest.raises(ValueError):
        chunk_text("a b", 10, 10)


def test_normalize_and_hash():
    assert normalize_text("  a \n\n b\t c ") == "a b c"
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hello!")
