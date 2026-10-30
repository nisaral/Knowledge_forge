"""Whole-site crawler against a local HTTP server (no internet)."""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from knowledge_forge import config, crawler, ingestion
from knowledge_forge.web import create_app

WORDS = "Photosynthesis converts light energy into chemical energy inside chloroplasts. "


def page(title, body, links=(), extra_head=""):
    a = "".join(f'<a href="{h}">{h}</a> ' for h in links)
    return (f"<html><head><title>{title}</title>{extra_head}</head><body><main><h1>{title}</h1>"
            f"<p>{body}</p>{a}</main></body></html>").encode()


def minimal_pdf(text: str) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


class Handler(BaseHTTPRequestHandler):
    hits: list = []
    routes: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        Handler.hits.append(self.path)
        route = Handler.routes.get(self.path.split("#")[0])
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        status, headers, body = route
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTML = {"Content-Type": "text/html; charset=utf-8"}


@pytest.fixture
def site(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    long = WORDS * 6
    Handler.hits = []
    Handler.routes = {
        "/robots.txt": (200, {"Content-Type": "text/plain"},
                        b"User-agent: *\nDisallow: /private\nCrawl-delay: 0\n"
                        + f"Sitemap: {base}/sitemap.xml\n".encode()),
        "/sitemap.xml": (200, {"Content-Type": "application/xml"},
                         f"<?xml version='1.0'?><urlset><url><loc>{base}/orphan</loc></url>"
                         f"<url><loc>https://elsewhere.example/x</loc></url></urlset>".encode()),
        "/": (200, HTML, page("Home", long, ["/a", "/b?utm_source=x", "/private/secret", "/doc.pdf",
                                             "/dup", "/evil", "/offsite", "mailto:x@y.z", "/img.png"])),
        "/a": (200, HTML, page("Page A", "Alpha " + long, ["/", "/b", "/a#frag", "/deep1"])),
        "/b": (200, HTML, page("Page B", "Bravo " + long, ["/a", "/b/../a"])),
        "/deep1": (200, HTML, page("Deep 1", "Deep one " + long, ["/deep2"])),
        "/deep2": (200, HTML, page("Deep 2", "Deep two " + long, ["/deep3"])),
        "/deep3": (200, HTML, page("Deep 3", "Deep three " + long)),
        "/orphan": (200, HTML, page("Orphan", "Only in the sitemap. " + long)),
        "/private/secret": (200, HTML, page("Secret", "should not be fetched " + long)),
        "/doc.pdf": (200, {"Content-Type": "application/pdf"},
                     minimal_pdf("Calvin cycle fixes carbon dioxide into sugar in the stroma. " * 3)),
        "/evil": (302, {"Location": "http://10.255.255.1/meta"}, b""),
        "/offsite": (302, {"Location": "https://elsewhere.example/"}, b""),
    }
    Handler.routes["/dup"] = Handler.routes["/"]  # same content under another URL
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # 127.0.0.1 stands in for "the public internet"; 10/8, 169.254/16 and 192.168/16 stay blocked.
    monkeypatch.setattr(ingestion, "is_blocked_ip", lambda ip: str(ip).startswith(("10.", "169.254.", "192.168.")))
    monkeypatch.setenv("CRAWL_RESPECT_ROBOTS", "true")
    monkeypatch.setenv("CRAWL_SITE_DELAY_SECONDS", "0")
    config.get_settings.cache_clear()
    ingestion._ROBOTS_CACHE.clear()
    yield base
    server.shutdown()


def run_job(kb, url, **opts):
    defaults = dict(max_pages=50, max_depth=3, include_subdomains=False, use_sitemap=True, render_js="never")
    defaults.update(opts)
    job = crawler.CrawlJob(kb, url, **defaults)
    job.run()
    return job


def test_canonicalize():
    c = crawler.canonicalize
    assert c("HTTP://Example.COM:80/a/./b/../c?b=2&a=1&utm_source=x#frag") == "http://example.com/a/c?a=1&b=2"
    assert c("https://example.com") == "https://example.com/"
    assert c("https://example.com:8443/x/") == "https://example.com:8443/x/"
    assert c("/rel", "https://example.com/dir/page") == "https://example.com/rel"
    assert c("javascript:alert(1)") is None and c("ftp://x/") is None


def test_scope_and_registered_domain():
    assert crawler.registered_domain("a.b.example.co.uk") == "example.co.uk"
    assert crawler.registered_domain("docs.python.org") == "python.org"
    assert crawler.in_scope("https://www.example.com/x", "example.com", False)
    assert not crawler.in_scope("https://docs.example.com/x", "example.com", False)
    assert crawler.in_scope("https://docs.example.com/x", "www.example.com", True)
    assert not crawler.in_scope("https://example.org/x", "example.com", True)
    assert not crawler.in_scope("https://example.com/blog/x", "example.com", False, "/docs/")


def test_parse_sitemap_index_and_gzip():
    import gzip

    locs, idx = crawler.parse_sitemap(b"<sitemapindex><sitemap><loc>https://e.com/s1.xml</loc></sitemap></sitemapindex>")
    assert idx and locs == ["https://e.com/s1.xml"]
    locs, idx = crawler.parse_sitemap(gzip.compress(b"<urlset><url><loc>https://e.com/a?x=1&amp;y=2</loc></url></urlset>"))
    assert not idx and locs == ["https://e.com/a?x=1&y=2"]


def test_full_crawl(site, kb):
    job = run_job(kb, site + "/")
    assert job.status == "done"
    pages = job.pages
    st = {u.replace(site, ""): p["status"] for u, p in pages.items()}
    # robots.txt Disallow respected, and the disallowed page never requested
    assert st["/private/secret"] == "skipped"
    assert not any(h.startswith("/private") for h in Handler.hits)
    # loops, fragments, tracking params and dot-segments all collapse to one fetch each
    assert Handler.hits.count("/a") == 1 and Handler.hits.count("/b") == 1
    assert "/b?utm_source=x" not in st
    # sitemap discovery adds orphan pages; off-site sitemap entries ignored
    assert st["/orphan"] == "ingested"
    assert job.sitemap_urls == 1
    # PDF ingested
    assert st["/doc.pdf"] == "ingested" and pages[site + "/doc.pdf"]["kind"] == "pdf"
    # redirect to a private IP blocked; off-site redirect skipped
    assert st["/evil"] == "failed" and "private" in pages[site + "/evil"]["error"]
    assert st["/offsite"] == "skipped"
    # content duplicate skipped
    assert st["/dup"] == "skipped" and "Duplicate" in pages[site + "/dup"]["error"]
    # depth limit: / (0) -> /a (1) -> /deep1 (2) -> /deep2 (3) -> /deep3 (4, not enqueued)
    assert st["/deep2"] == "ingested" and "/deep3" not in st
    sources = {s["source"]: s for s in kb.list_sources()}
    assert sources[site + "/a"]["group"] == "site:127.0.0.1"
    assert sources[site + "/a"]["title"] == "Page A"
    assert job.counts["ingested"] == len([s for s in st.values() if s == "ingested"])
    assert job.counts["failed"] >= 1

    removed, n_sources = kb.delete_group("site:127.0.0.1")
    assert removed > 0 and n_sources == job.counts["ingested"]
    assert kb.list_sources() == []


def test_max_pages_and_depth_zero(site, kb):
    job = run_job(kb, site + "/", max_pages=2)
    assert job._dispatched == 2
    job = run_job(kb, site + "/a", max_depth=0)
    assert [p["status"] for p in job.pages.values()] == ["ingested"]


def test_crawl_delay_from_robots(site, kb):
    Handler.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"}, b"User-agent: *\nCrawl-delay: 1\n")
    ingestion._ROBOTS_CACHE.clear()
    t0 = time.monotonic()
    job = run_job(kb, site + "/b", max_depth=1, use_sitemap=False)
    assert job.robots_delay == pytest.approx(1.0)
    assert job.counts["fetched"] == 2 and time.monotonic() - t0 >= 0.9


def test_robots_disallow_all(site, kb):
    Handler.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"}, b"User-agent: *\nDisallow: /\n")
    ingestion._ROBOTS_CACHE.clear()
    job = run_job(kb, site + "/")
    assert job.counts["fetched"] == 0 and "robots" in job.error
    assert [h for h in Handler.hits if h != "/robots.txt"] == []


def test_meta_noindex_nofollow(site, kb):
    Handler.routes["/"] = (200, HTML, page("Home", WORDS * 6, ["/a"],
                                         '<meta name="robots" content="noindex, nofollow">'))
    job = run_job(kb, site + "/", use_sitemap=False)
    assert job.pages[site + "/"]["status"] == "skipped" and len(job.pages) == 1


def test_cancel_and_retry(site, kb, monkeypatch):
    monkeypatch.setenv("CRAWL_SITE_DELAY_SECONDS", "0.4")
    config.get_settings.cache_clear()
    job = crawler.CrawlJob(kb, site + "/", max_pages=50, max_depth=3, include_subdomains=False,
                           use_sitemap=False, render_js="never")
    job.start()
    time.sleep(0.6)
    job.cancel()
    job.join(10)
    assert job.status == "cancelled"
    assert any(p["status"] == "not_fetched" for p in job.pages.values())
    monkeypatch.setenv("CRAWL_SITE_DELAY_SECONDS", "0")
    config.get_settings.cache_clear()
    assert job.retry() > 0
    job.join(20)
    assert job.status == "done"
    assert not any(p["status"] == "not_fetched" for p in job.pages.values())


def test_crawl_api(site, kb):
    app = create_app(kb)
    c = app.test_client()
    assert c.post("/api/crawl", json={"url": "http://10.0.0.5/"}).status_code == 400
    assert c.post("/api/crawl", json={"url": site, "max_pages": 0}).status_code == 400
    assert c.post("/api/crawl", json={"url": site, "include_subdomains": "yes"}).status_code == 400
    r = c.post("/api/crawl", json={"url": site + "/", "max_pages": 5, "max_depth": 1})
    assert r.status_code == 202
    jid = r.get_json()["job"]["id"]
    for _ in range(100):
        job = c.get(f"/api/crawl/{jid}").get_json()["job"]
        if job["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.1)
    assert job["status"] == "done" and job["counts"]["ingested"] >= 2
    assert any(j["id"] == jid for j in c.get("/api/crawl").get_json()["jobs"])
    assert c.get("/api/crawl/nope").status_code == 404
    src = c.get("/api/get-sources").get_json()["sources"]
    assert all(s["group"] == "site:127.0.0.1" for s in src)
    r = c.post("/api/delete-group", json={"group": "site:127.0.0.1"})
    assert r.status_code == 200 and r.get_json()["removed"] > 0
    assert c.post("/api/delete-group", json={"group": "site:127.0.0.1"}).status_code == 404
