import socket

import pytest
import requests

from knowledge_forge import ingestion
from knowledge_forge.errors import ConfigurationError, IngestionError


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
    monkeypatch.setattr(ingestion, "crawl_url_detailed", lambda u, *_a: (f"content of {u}", ""))
    text, err, stype = ingestion.ingest_text_source("example.com", "web")
    assert text == "content of https://example.com" and stype == "web" and not err


def test_ingest_web_rejects_private(monkeypatch):
    monkeypatch.setattr(ingestion, "validate_public_url", lambda _u: "nope")
    assert ingestion.ingest_text_source("http://10.0.0.1", "web") == (None, "nope", "web")


def test_youtube_link_is_detected_from_web(monkeypatch):
    monkeypatch.setattr(ingestion, "get_youtube_transcript", lambda u: ("transcript", ""))
    assert ingestion.ingest_text_source("https://youtu.be/dQw4w9WgXcQ", "web")[2] == "youtube"


class _Resp:
    def __init__(self, body=b"", status=200, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers if headers is not None else {"content-type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"
        self.closed = False

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def close(self):
        self.closed = True


def test_crawl_url_extracts_main_text(monkeypatch):
    html = "<html><body><nav>menu</nav><article><p>" + "Useful words here. " * 20 + "</p></article></body></html>"
    monkeypatch.setattr(ingestion, "validate_public_url", lambda _u: None)
    monkeypatch.setattr(ingestion, "_http_get", lambda *_a, **_k: _Resp(html.encode()))
    text = ingestion.crawl_url("https://example.com")
    assert text and "Useful words here." in text and "menu" not in text


def test_crawl_url_blocks_redirect_to_private(monkeypatch):
    redirect = _Resp(status=302, headers={"location": "http://169.254.169.254/latest"})
    monkeypatch.setattr(ingestion, "validate_public_url", lambda u: "private" if "169.254" in u else None)
    monkeypatch.setattr(ingestion, "_http_get", lambda *_a, **_k: redirect)
    assert ingestion.crawl_url("https://example.com") is None
    assert redirect.closed


def test_crawl_rejects_oversized_and_binary(monkeypatch):
    monkeypatch.setenv("CRAWL_MAX_BYTES", "10000")
    from knowledge_forge import config

    config.get_settings.cache_clear()
    monkeypatch.setattr(ingestion, "validate_public_url", lambda _u: None)
    monkeypatch.setattr(ingestion, "_http_get", lambda *_a, **_k: _Resp(b"x" * 50000))
    text, reason = ingestion.crawl_url_detailed("https://example.com")
    assert text is None and "limit" in reason
    pdf = _Resp(b"%PDF-1.4", headers={"content-type": "application/pdf"})
    monkeypatch.setattr(ingestion, "_http_get", lambda *_a, **_k: pdf)
    text, reason = ingestion.crawl_url_detailed("https://example.com/a.pdf")
    assert text is None and reason


def test_crawl_too_many_redirects(monkeypatch):
    monkeypatch.setattr(ingestion, "validate_public_url", lambda _u: None)
    monkeypatch.setattr(ingestion, "_http_get",
                        lambda *_a, **_k: _Resp(status=301, headers={"location": "/again"}))
    assert ingestion.crawl_url_detailed("https://example.com")[0] is None


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.1.2.3", "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1",
                                "fe80::1", "fc00::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "224.0.0.1",
                                "2002:7f00:0001::", "not-an-ip"])
def test_is_blocked_ip_blocks_non_public(ip):
    assert ingestion.is_blocked_ip(ip)


@pytest.mark.parametrize("ip", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946", "8.8.8.8"])
def test_is_blocked_ip_allows_public(ip):
    assert not ingestion.is_blocked_ip(ip)


@pytest.mark.parametrize("url", ["http://user:pw@example.com", "http://example.com:99999", "http://printer.local",
                                 "http://db.internal", "file:///etc/passwd", "gopher://example.com",
                                 "http://[::1]/"])
def test_validate_public_url_rejects_tricks(url, monkeypatch):
    monkeypatch.setattr(ingestion.socket, "getaddrinfo", _resolve_to("93.184.216.34"))
    assert ingestion.validate_public_url(url) is not None


def test_guarded_connection_refuses_rebound_private_address(monkeypatch):
    """DNS rebinding: the pre-check saw a public IP, the connect-time lookup returns a private one."""
    monkeypatch.setattr(ingestion.socket, "getaddrinfo", _resolve_to("127.0.0.1"))
    with pytest.raises(OSError):
        ingestion._guarded_create_connection(("example.com", 80), timeout=1)


def test_sniff_file_type_checks_magic_bytes():
    assert ingestion.sniff_file_type("a.pdf", "application/pdf", b"%PDF-1.7 ...")[0] == "pdf"
    assert ingestion.sniff_file_type("a.pdf", "application/pdf", b"MZ\x90\x00evil") is None
    assert ingestion.sniff_file_type("a.png", "image/png", b"\x89PNG\r\n\x1a\n....")[0] == "image"
    assert ingestion.sniff_file_type("a.txt", "text/plain", b"hello") == ("text", "text/plain") or \
        ingestion.sniff_file_type("a.txt", "text/plain", b"hello")[0] == "text"
    assert ingestion.sniff_file_type("a.txt", "text/plain", b"he\x00llo") is None
    assert ingestion.sniff_file_type("a.exe", "application/octet-stream", b"MZ") is None


def test_safe_filename_strips_paths_and_controls():
    assert ingestion.safe_filename("../../etc/passwd") == "passwd"
    assert ingestion.safe_filename("C:\\x\\evil\x00.txt") == "evil.txt"
    assert ingestion.safe_filename("") == "upload"
    assert len(ingestion.safe_filename("a" * 500 + ".pdf")) <= 120


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


# ── regressions from the live end-to-end run (docs/TESTING.md) ─────────────
def test_short_text_pdf_kept_when_ocr_unavailable(monkeypatch):
    def no_key(_d):
        raise ConfigurationError("no key")

    monkeypatch.setattr(ingestion, "ingest_pdf_via_gemini_bytes", no_key)
    assert ingestion.ingest_pdf_bytes(_make_pdf("Dummy PDF file")) == "Dummy PDF file"
    monkeypatch.setattr(ingestion, "ingest_pdf_via_gemini_bytes", lambda _d: None)
    assert ingestion.ingest_pdf_bytes(_make_pdf("Dummy PDF file")) == "Dummy PDF file"


def test_image_only_pdf_without_ocr_has_clear_error(monkeypatch):
    monkeypatch.setattr(ingestion, "ingest_pdf_via_gemini_bytes", lambda _d: None)
    with pytest.raises(IngestionError, match="no text layer"):
        ingestion.ingest_pdf_bytes(_make_pdf(""))


def test_corrupt_pdf_has_clear_error(monkeypatch):
    monkeypatch.setattr(ingestion, "ingest_pdf_via_gemini_bytes", lambda _d: None)
    with pytest.raises(IngestionError, match="damaged or truncated"):
        ingestion.ingest_pdf_bytes(_make_pdf("x" * 500)[:300])


def _static(monkeypatch, result):
    monkeypatch.setattr(ingestion, "validate_public_url", lambda u: None)
    monkeypatch.setattr(ingestion, "_fetch_static", lambda u: result)


def test_auto_mode_does_not_render_after_size_or_http_error(monkeypatch):
    from knowledge_forge import config, js_render

    monkeypatch.setenv("RENDER_JS", "auto")
    config.get_settings.cache_clear()
    monkeypatch.setattr(js_render, "render_page", lambda u: pytest.fail("should not render"))
    for reason in ("The page is larger than the 5120 KB crawl limit.", "The site responded with HTTP 404.",
                   "The site took too long to respond."):
        _static(monkeypatch, (None, "", "", reason))
        assert ingestion.crawl_url_detailed("https://big.example") == (None, reason)


def test_sitemap_xml_is_rejected_with_hint(monkeypatch):
    body = (b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            b"<url><loc>https://a/</loc></url></urlset>")
    _static(monkeypatch, (body, "application/xml", "utf-8", ""))
    text, err = ingestion.crawl_url_detailed("https://example.com/sitemap.xml")
    assert text is None and "sitemap" in err


class _RobotsResp:
    def __init__(self, status, body=b""):
        self.status_code, self._body, self.headers = status, body, {}

    def iter_content(self, chunk_size=1):
        yield self._body

    def close(self):
        pass


def test_robots_txt_respected(monkeypatch):
    from knowledge_forge import config

    monkeypatch.setenv("CRAWL_RESPECT_ROBOTS", "true")
    config.get_settings.cache_clear()
    ingestion._ROBOTS_CACHE.clear()
    robots = b"User-agent: *\nDisallow: /wiki/Special:\n"
    calls = []

    def fake_get(url, **kw):
        calls.append(kw["headers"]["User-Agent"])
        return _RobotsResp(200, robots)

    monkeypatch.setattr(ingestion, "_http_get", fake_get)
    _static(monkeypatch, (b"<p>" + b"Allowed article text. " * 10 + b"</p>", "text/html", "utf-8", ""))
    blocked = ingestion.crawl_url_detailed("https://w.example/wiki/Special:Search?q=x")
    assert blocked == (None, ingestion.ROBOTS_MESSAGE)
    assert ingestion.crawl_url_detailed("https://w.example/wiki/Photosynthesis")[0]
    assert calls == [config.get_settings().crawl_user_agent]  # sent our UA, fetched once (cached)


def test_robots_txt_missing_or_unreachable_allows(monkeypatch):
    from knowledge_forge import config

    monkeypatch.setenv("CRAWL_RESPECT_ROBOTS", "true")
    config.get_settings.cache_clear()
    ingestion._ROBOTS_CACHE.clear()
    monkeypatch.setattr(ingestion, "_http_get", lambda url, **kw: _RobotsResp(404))
    assert ingestion.robots_allowed("https://a.example/x")

    def boom(url, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(ingestion, "_http_get", boom)
    assert ingestion.robots_allowed("https://b.example/x")


def test_accept_header_has_wildcard_fallback(monkeypatch):
    # sitemaps.org answered 406 to an Accept header without */* during the live run.
    seen = {}

    def fake_get(url, **kw):
        seen.update(kw["headers"])
        return _RobotsResp(404)

    monkeypatch.setattr(ingestion, "_http_get", fake_get)
    ingestion._fetch_static("https://example.com/")
    assert "*/*" in seen["Accept"]


def test_truncated_but_openable_pdf_reports_damage(monkeypatch):
    import pymupdf

    monkeypatch.setattr(ingestion, "ingest_pdf_via_gemini_bytes", lambda _d: None)
    doc = pymupdf.open()
    doc.new_page()  # no text, so extraction is empty
    data = doc.tobytes()
    with pytest.raises(IngestionError, match="damaged"):
        ingestion.ingest_pdf_bytes(data[: len(data) * 2 // 3])
