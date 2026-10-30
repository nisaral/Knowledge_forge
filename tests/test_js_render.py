"""JavaScript rendering: mode handling, fallbacks and a real headless-browser run against a local server."""
import builtins
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

from knowledge_forge import config, ingestion, js_render
from knowledge_forge.errors import IngestionError

LONG = "Rendered paragraph about photosynthesis and chloroplasts. " * 20
SHELL = b'<html><body><div id="root"></div><script src="/app.js"></script></body></html>'


def _settings(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config.get_settings.cache_clear()


# ── unit tests (no browser) ────────────────────────────────────────────────
def test_mode_defaults_to_never_and_invalid_falls_back(monkeypatch):
    assert config.get_settings().render_js == "never"
    _settings(monkeypatch, RENDER_JS="sometimes")
    assert config.get_settings().render_js == "never"


def test_resolve_render_mode(monkeypatch):
    assert ingestion.resolve_render_mode(None) == "never"
    assert ingestion.resolve_render_mode("never") == "never"
    with pytest.raises(IngestionError, match="disabled"):
        ingestion.resolve_render_mode("always")
    _settings(monkeypatch, RENDER_JS="auto")
    assert ingestion.resolve_render_mode(None) == "auto"
    assert ingestion.resolve_render_mode("always") == "always"
    with pytest.raises(IngestionError, match="one of"):
        ingestion.resolve_render_mode("yes")


def test_looks_like_spa(monkeypatch):
    assert js_render.looks_like_spa(SHELL, "")
    assert js_render.looks_like_spa(b'<div id="__next"></div><p>x</p>', "x" * 900)
    assert not js_render.looks_like_spa(b"<article>...</article>", "word " * 200)


def test_render_disabled_and_private_url_refused_before_launch(monkeypatch):
    with pytest.raises(IngestionError, match="disabled"):
        js_render.render_page("https://example.com")
    _settings(monkeypatch, RENDER_JS="always")
    with pytest.raises(IngestionError, match="private"):
        js_render.render_page("http://127.0.0.1/")


def test_missing_playwright_gives_install_hint(monkeypatch):
    _settings(monkeypatch, RENDER_JS="always")
    monkeypatch.setattr(js_render, "_check_url", lambda u: None)
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("playwright"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(IngestionError, match="playwright install chromium"):
        js_render.render_page("https://example.com")


def _stub_static(monkeypatch, body, ctype="text/html", reason=""):
    monkeypatch.setattr(ingestion, "validate_public_url", lambda u: None)
    monkeypatch.setattr(ingestion, "_fetch_static", lambda u: (body, ctype, "utf-8", reason))


def test_auto_renders_spa_shell(monkeypatch):
    _settings(monkeypatch, RENDER_JS="auto")
    _stub_static(monkeypatch, SHELL)
    calls = []
    monkeypatch.setattr(js_render, "render_page", lambda u: calls.append(u) or (f"<main>{LONG}</main>", u))
    text, err = ingestion.crawl_url_detailed("https://spa.example")
    assert err == "" and "Rendered paragraph" in text and calls


def test_auto_keeps_static_text_when_render_fails(monkeypatch):
    _settings(monkeypatch, RENDER_JS="auto")
    _stub_static(monkeypatch, b"<article>" + b"Static words here. " * 10 + b"</article>")

    def fail(u):
        raise IngestionError(js_render.INSTALL_HINT)

    monkeypatch.setattr(js_render, "render_page", fail)
    text, err = ingestion.crawl_url_detailed("https://short.example")
    assert text and "Static words" in text


def test_auto_reports_install_hint_when_nothing_usable(monkeypatch):
    _settings(monkeypatch, RENDER_JS="auto")
    _stub_static(monkeypatch, SHELL)
    def fail(u):
        raise IngestionError(js_render.INSTALL_HINT)

    monkeypatch.setattr(js_render, "render_page", fail)
    text, err = ingestion.crawl_url_detailed("https://spa.example")
    assert text is None and "Playwright" in err


def test_never_does_not_render_and_always_skips_static(monkeypatch):
    _stub_static(monkeypatch, SHELL)
    monkeypatch.setattr(js_render, "render_page", lambda u: pytest.fail("rendered"))
    text, err = ingestion.crawl_url_detailed("https://spa.example")
    assert text is None and "JavaScript" in err
    _settings(monkeypatch, RENDER_JS="always")
    monkeypatch.setattr(ingestion, "_fetch_static", lambda u: pytest.fail("static fetch"))
    monkeypatch.setattr(js_render, "render_page", lambda u: (f"<p>{LONG}</p>", u))
    assert ingestion.crawl_url_detailed("https://spa.example")[0]


def test_add_content_passes_render_js_and_get_sources_reports_status(monkeypatch, fake_embed):
    from knowledge_forge import web

    seen = {}
    monkeypatch.setattr(web, "ingest_text_source", lambda s, t, r=None: seen.update(r=r) or (LONG, "", "web"))
    client = web.create_app().test_client()
    r = client.post("/api/add-content", json={"source": "https://e.com", "source_type": "web", "render_js": "auto"})
    assert r.status_code == 200 and seen["r"] == "auto"
    assert client.get("/api/get-sources").get_json()["render_js"] == {"mode": "never", "available": False}


def test_concurrency_cap(monkeypatch):
    _settings(monkeypatch, RENDER_JS="always", RENDER_JS_MAX_CONCURRENCY="1", RENDER_JS_TIMEOUT_SECONDS="2")
    monkeypatch.setattr(js_render, "_check_url", lambda u: None)
    fake = types.ModuleType("playwright.sync_api")
    fake.sync_playwright = lambda: pytest.fail("browser launched while busy")
    fake.Error = fake.TimeoutError = RuntimeError
    monkeypatch.setitem(sys.modules, "playwright", types.ModuleType("playwright"))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake)
    sem = js_render._semaphore()
    sem.acquire()
    try:
        with pytest.raises(IngestionError, match="busy"):
            js_render.render_page("https://example.com")
    finally:
        sem.release()


# ── real browser against a local server ────────────────────────────────────
PAGES = {
    "/spa": SHELL,
    "/app.js": b"fetch('/api/data.json').then(r=>r.json()).then(d=>{"
               b"document.getElementById('root').innerHTML='<main><h1>'+d.title+'</h1><p>'+d.body+'</p></main>';});",
    "/lazy": b"<html><body><main id=m></main><script>setTimeout(()=>{document.getElementById('m').textContent="
             b"'Lazy loaded content arrived after a delay. '.repeat(20)},300)</script></body></html>",
    "/leak": b"<html><body><main><p id=p>" + b"Visible public text. " * 10 + b"</p></main><script>"
             b"fetch('http://localhost:PORT/secret').then(r=>r.text()).then(t=>{document.getElementById('p')"
             b".textContent+=' LEAKED:'+t}).catch(()=>{});</script></body></html>",
}


class _Handler(BaseHTTPRequestHandler):
    hits: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        _Handler.hits.append((self.headers.get("Host", ""), path))
        if path == "/hop":
            self.send_response(302)
            self.send_header("Location", "/spa")
            self.end_headers()
            return
        if path == "/redirect":
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_port}/secret")
            self.end_headers()
            return
        if path == "/api/data.json":
            body, ctype = (b'{"title":"SPA Title","body":"' + LONG.encode() + b'"}', "application/json")
        elif path == "/secret":
            body, ctype = b"internal-secret", "text/plain"
        elif path in PAGES:
            body = PAGES[path].replace(b"PORT", str(self.server.server_port).encode())
            ctype = "application/javascript" if path.endswith(".js") else "text/html"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _browser_ok() -> bool:
    if not js_render.playwright_installed():
        return False
    try:
        import os

        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            path = os.environ.get("RENDER_JS_BROWSER_PATH")
            kw = {"executable_path": path} if path else {}
            pw.chromium.launch(**kw).close()
        return True
    except Exception:
        return False


browser = pytest.mark.skipif(not _browser_ok(), reason="Playwright/Chromium not installed")


@pytest.fixture
def server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Handler.hits = []
    # 127.0.0.1 stands in for "the public internet"; "localhost" stands in for a private address.
    _settings(monkeypatch, RENDER_JS="auto", ALLOW_PRIVATE_URLS="true", RENDER_JS_SETTLE_MS="800")
    monkeypatch.setattr(js_render, "_check_url",
                        lambda u: "private" if urlparse(u).hostname == "localhost" else None)
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@browser
def test_real_render_spa(server):
    text, err = ingestion.crawl_url_detailed(server + "/spa")
    assert err == "" and "SPA Title" in text and "Rendered paragraph" in text


@browser
def test_real_render_lazy_content(server):
    html, _ = js_render.render_page(server + "/lazy")
    assert "Lazy loaded content" in html


@browser
def test_real_render_blocks_private_subrequest(server):
    html, _ = js_render.render_page(server + "/leak")
    assert "Visible public text" in html and "LEAKED:internal" not in html
    assert not any(path == "/secret" for _host, path in _Handler.hits)


@browser
def test_real_render_follows_public_redirect(server):
    _Handler.hits = []
    html, _ = js_render.render_page(server + "/hop")
    assert "SPA Title" in html


@browser
def test_real_render_blocks_private_redirect(server):
    with pytest.raises(IngestionError, match="private"):
        js_render.render_page(server + "/redirect")
    assert not any(path == "/secret" for _host, path in _Handler.hits)
