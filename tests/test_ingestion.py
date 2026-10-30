import socket

import pytest

from knowledge_forge import ingestion
from knowledge_forge.errors import IngestionError


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", True),
        ("https://youtu.be/dQw4w9WgXcQ", True),
        ("https://notyoutube.com/watch?v=abc", False),
        ("https://example.com/youtube.com", False),
    ],
)
def test_is_youtube_url(url, expected):
    assert ingestion.is_youtube_url(url) is expected


def test_extract_youtube_id():
    assert ingestion._extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1") == "dQw4w9WgXcQ"
    assert ingestion._extract_youtube_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_normalize_url_adds_scheme():
    assert ingestion.normalize_url("example.com/a") == "https://example.com/a"


def _resolve_to(ip):
    return lambda host, *_a, **_k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


def test_validate_public_url_blocks_private(monkeypatch):
    assert ingestion.validate_public_url("ftp://example.com") is not None
    assert ingestion.validate_public_url("http://localhost:5000") is not None
    monkeypatch.setattr(ingestion.socket, "getaddrinfo", _resolve_to("10.0.0.5"))
    assert ingestion.validate_public_url("http://internal.example") is not None
    monkeypatch.setattr(ingestion.socket, "getaddrinfo", _resolve_to("93.184.216.34"))
    assert ingestion.validate_public_url("https://example.com") is None


def test_ingest_text_source_text_is_never_treated_as_url():
    text, err, stype = ingestion.ingest_text_source("see https://youtu.be/x for more", "text")
    assert (text, err, stype) == ("see https://youtu.be/x for more", "", "text")


def test_ingest_web_uses_crawler(monkeypatch):
    monkeypatch.setattr(ingestion, "validate_public_url", lambda _u: None)
    monkeypatch.setattr(ingestion, "crawl_url", lambda u: f"content of {u}")
    text, err, stype = ingestion.ingest_text_source("example.com", "web")
    assert text == "content of https://example.com" and stype == "web" and not err


def test_ingest_web_rejects_private(monkeypatch):
    monkeypatch.setattr(ingestion, "validate_public_url", lambda _u: "nope")
    assert ingestion.ingest_text_source("http://10.0.0.1", "web") == (None, "nope", "web")


def test_youtube_link_is_detected_from_web(monkeypatch):
    monkeypatch.setattr(ingestion, "get_youtube_transcript", lambda u: ("transcript", ""))
    assert ingestion.ingest_text_source("https://youtu.be/dQw4w9WgXcQ", "web")[2] == "youtube"


def test_crawl_url_extracts_main_text(monkeypatch):
    html = "<html><body><nav>menu</nav><article><p>" + "Useful words here. " * 20 + "</p></article></body></html>"

    class Resp:
        status_code = 200
        text = html
        content = html.encode()
        headers = {"content-type": "text/html"}
        encoding = "utf-8"
        apparent_encoding = "utf-8"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ingestion, "validate_public_url", lambda _u: None)
    monkeypatch.setattr(ingestion.requests, "get", lambda *_a, **_k: Resp())
    text = ingestion.crawl_url("https://example.com")
    assert text and "Useful words here." in text and "menu" not in text


def test_crawl_url_blocks_redirect_to_private(monkeypatch):
    class Redirect:
        status_code = 302
        headers = {"location": "http://169.254.169.254/latest"}

    monkeypatch.setattr(ingestion, "validate_public_url", lambda u: "private" if "169.254" in u else None)
    monkeypatch.setattr(ingestion.requests, "get", lambda *_a, **_k: Redirect())
    assert ingestion.crawl_url("https://example.com") is None


def _make_pdf(text="", password=None):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    kwargs = {}
    if password:
        kwargs = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "user_pw": password, "owner_pw": password}
    data = doc.tobytes(**kwargs)
    doc.close()
    return data


def test_pdf_text_extraction():
    data = _make_pdf("Knowledge forge test document. " * 6)
    assert "Knowledge forge test document." in ingestion.ingest_pdf_bytes(data)


def test_scanned_pdf_falls_back_to_gemini(monkeypatch):
    monkeypatch.setattr(ingestion, "ingest_pdf_via_gemini_bytes", lambda _d: "ocr text")
    assert ingestion.ingest_pdf_bytes(_make_pdf("")) == "ocr text"


def test_password_pdf_raises():
    with pytest.raises(IngestionError):
        ingestion.ingest_pdf_bytes(_make_pdf("secret", password="pw"))


@pytest.mark.parametrize(
    "name, mime, expected",
    [
        ("a.pdf", "", "pdf"),
        ("a.MP3", "", "audio"),
        ("a.mov", "", "video"),
        ("a.png", "", "image"),
        ("notes.md", "", "text"),
        ("a.exe", "application/octet-stream", None),
    ],
)
def test_detect_file_type(name, mime, expected):
    result = ingestion.detect_file_type(name, mime)
    assert (result[0] if result else None) == expected


@pytest.mark.network
def test_live_youtube_transcript():
    text, _detail = ingestion.get_youtube_transcript("https://www.youtube.com/watch?v=jNQXAC9IVRw")
    assert text
