"""Whole-website crawler: polite, SSRF-guarded BFS ingestion run as a background job.

* Scope: the start host (``www.`` and bare host are treated as the same site). With
  ``include_subdomains`` every host under the start URL's registered domain is in scope.
* Respects robots.txt (``Disallow`` and ``Crawl-delay``) when CRAWL_RESPECT_ROBOTS is on, and
  ``<meta name="robots">`` noindex / nofollow. Seeds from sitemap.xml and robots ``Sitemap:``.
* URLs are canonicalised (fragment, default port, tracking params, query order) and pages
  are de-duplicated by canonical URL, ``<link rel="canonical">`` and content hash.
* Every hop, including each redirect, is re-validated against the public-address check
  (and the HTTP client re-checks the connected IP).
* One request at a time per host is spaced by ``max(CRAWL_SITE_DELAY_SECONDS, Crawl-delay)``;
  CRAWL_SITE_CONCURRENCY bounds parallel fetches across hosts.
* Jobs live in process memory: poll the worker that started them (one gunicorn worker with
  the local store; see docs).
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import posixpath
import re
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

from knowledge_forge import ingestion
from knowledge_forge.chunking import chunk_text, normalize_text
from knowledge_forge.config import get_settings
from knowledge_forge.errors import IngestionError, NotFoundError, ValidationError

logger = logging.getLogger(__name__)

_TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|dclid|msclkid|mc_cid|mc_eid|_ga|_gl|ref_src|igshid|yclid)$", re.I)
_SKIP_EXT = re.compile(
    r"\.(?:png|jpe?g|gif|webp|svg|ico|bmp|tiff?|mp[34]|m4a|wav|ogg|webm|avi|mov|mkv|zip|gz|tgz|tar|rar|7z|"
    r"exe|dmg|msi|apk|iso|css|js|mjs|json|woff2?|ttf|eot|otf|docx?|xlsx?|pptx?|csv|rss|atom)$",
    re.I,
)
_LOC_RE = re.compile(rb"<(?:\w+:)?loc>\s*(.*?)\s*</(?:\w+:)?loc>", re.I | re.S)
_MULTI_SUFFIX = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "ltd.uk", "me.uk", "net.uk", "co.in", "org.in", "ac.in", "gov.in",
    "net.in", "edu.in", "co.jp", "ac.jp", "or.jp", "ne.jp", "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.nz", "org.nz", "ac.nz", "com.br", "com.cn", "edu.cn", "gov.cn", "com.mx", "co.za", "ac.za", "com.sg",
    "edu.sg", "com.hk", "co.kr", "or.kr", "com.tr", "com.tw", "edu.tw", "github.io", "gitlab.io", "pages.dev",
    "netlify.app", "vercel.app", "herokuapp.com", "readthedocs.io", "blogspot.com", "wordpress.com",
}
PDF_TYPES = ("application/pdf", "application/x-pdf")
PAGE_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
SITEMAP_TYPES = ("application/xml", "text/xml", "text/plain", "application/x-gzip", "application/gzip",
                 "application/octet-stream", "text/html")


# ─── URL helpers ────────────────────────────────────────────────────────────
def canonicalize(url: str, base: str | None = None) -> str | None:
    """Canonical absolute http(s) URL, or None if it is not crawlable."""
    url = (url or "").strip()
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    p = urlparse(url)
    if p.scheme.lower() not in ("http", "https") or not p.hostname:
        return None
    scheme = p.scheme.lower()
    host = p.hostname.lower().rstrip(".")
    try:
        port = p.port
    except ValueError:
        return None
    netloc = host if port is None or (scheme, port) in (("http", 80), ("https", 443)) else f"{host}:{port}"
    path = p.path or "/"
    path = re.sub(r"/{2,}", "/", path)
    norm = posixpath.normpath(path)
    if path.endswith("/") and norm != "/":
        norm += "/"
    path = norm if norm.startswith("/") else "/" + norm
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k)))
    return urlunparse((scheme, netloc, path, "", query, ""))


def registered_domain(host: str) -> str:
    """Best-effort registrable domain (no network-fetched suffix list): ``a.b.example.co.uk`` → ``example.co.uk``."""
    host = (host or "").lower().rstrip(".")
    if re.fullmatch(r"[\d.]+|\[?[0-9a-f:]+\]?", host):
        return host
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_SUFFIX:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _bare(host: str) -> str:
    host = (host or "").lower()
    return host[4:] if host.startswith("www.") else host


def in_scope(url: str, start_host: str, include_subdomains: bool, path_prefix: str = "/") -> bool:
    p = urlparse(url)
    host = (p.hostname or "").lower()
    if include_subdomains:
        rd = registered_domain(start_host)
        ok = host == rd or host.endswith("." + rd)
    else:
        ok = _bare(host) == _bare(start_host)
    return ok and (p.path or "/").startswith(path_prefix or "/")


def parse_html(body: bytes | str, base_url: str) -> dict:
    """Title, readable text, outgoing links, canonical URL and robots meta of an HTML page."""
    soup = BeautifulSoup(body, "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())[:200]
    if not title and soup.find("h1"):
        title = " ".join(soup.find("h1").get_text(" ").split())[:200]
    robots = ""
    for meta in soup.find_all("meta", attrs={"name": re.compile(r"^(robots|knowledgeforge)$", re.I)}):
        robots += " " + (meta.get("content") or "").lower()
    base_tag = soup.find("base", href=True)
    base = urljoin(base_url, base_tag["href"]) if base_tag else base_url
    canonical = None
    link = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]), href=True)
    if link:
        canonical = canonicalize(link["href"], base)
    links = []
    for a in soup.find_all("a", href=True):
        rel = a.get("rel") or []
        if "nofollow" in [r.lower() for r in rel]:
            continue
        href = a["href"].strip()
        if href.startswith(("mailto:", "javascript:", "tel:", "data:", "#")):
            continue
        u = canonicalize(href, base)
        if u:
            links.append(u)
    return {
        "title": title,
        "text": ingestion._html_to_text(body),
        "links": links,
        "canonical": canonical,
        "noindex": "noindex" in robots or "none" in robots.split(),
        "nofollow": "nofollow" in robots or "none" in robots.split(),
    }


def parse_sitemap(body: bytes) -> tuple[list[str], bool]:
    """``(locs, is_index)`` from a sitemap or sitemap index (regex parse: no XML entity expansion)."""
    if body[:2] == b"\x1f\x8b":
        try:
            body = gzip.decompress(body)[: 20 * 1024 * 1024]
        except OSError:
            return [], False
    is_index = bool(re.search(rb"<(?:\w+:)?sitemapindex[\s>]", body[:4096]))
    locs = []
    for m in _LOC_RE.finditer(body):
        raw = m.group(1).decode("utf-8", errors="replace")
        raw = raw.replace("&amp;", "&").replace("<![CDATA[", "").replace("]]>", "").strip()
        if raw:
            locs.append(raw)
    return locs, is_index


def store_text(kb, text: str, source: str, source_type: str, metadata: dict | None = None) -> dict:
    """Chunk and sync ``text`` as ``source``; shared by the API and the crawler."""
    settings = get_settings()
    text = normalize_text(text or "")
    if not text:
        raise IngestionError("No readable text could be extracted from this source.")
    truncated = len(text) > settings.max_text_chars
    text = text[: settings.max_text_chars]
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    added, removed = kb.sync_source(chunks, source, source_type, metadata=metadata, replace=True)
    return {"added": added, "removed": removed, "total": len(chunks), "truncated": truncated}


# ─── Jobs ───────────────────────────────────────────────────────────────────
class CrawlJob:
    def __init__(self, kb, url: str, *, max_pages: int, max_depth: int, include_subdomains: bool,
                 use_sitemap: bool, render_js: str, path_prefix: str = "/"):
        self.id = uuid.uuid4().hex[:12]
        self.kb = kb
        self.start_url = url
        p = urlparse(url)
        self.host = (p.hostname or "").lower()
        self.group = f"site:{_bare(self.host)}"
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.include_subdomains = include_subdomains
        self.use_sitemap = use_sitemap
        self.render_js = render_js
        self.path_prefix = path_prefix or "/"
        self.status = "queued"
        self.error = ""
        self.created = time.time()
        self.finished: float | None = None
        self.pages: dict[str, dict] = {}
        self.seen: set[str] = set()
        self.hashes: set[str] = set()
        self.counts = {"found": 0, "fetched": 0, "failed": 0, "ingested": 0, "skipped": 0, "chunks": 0}
        self.sitemap_urls = 0
        self.robots_delay = 0.0
        self._dispatched = 0
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._next_at: dict[str, float] = {}
        self._thread: threading.Thread | None = None

    # ── public view ──
    def snapshot(self, include_pages: bool = True) -> dict:
        with self._lock:
            out = {
                "id": self.id, "url": self.start_url, "group": self.group, "site": _bare(self.host),
                "status": self.status, "error": self.error, "counts": dict(self.counts),
                "max_pages": self.max_pages, "max_depth": self.max_depth,
                "include_subdomains": self.include_subdomains, "render_js": self.render_js,
                "sitemap_urls": self.sitemap_urls, "crawl_delay": round(self._delay(), 2),
                "created": self.created, "finished": self.finished,
            }
            if include_pages:
                out["pages"] = [dict(p) for p in self.pages.values()]
            return out

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            if self.status in ("queued", "running"):
                self.status = "cancelling"

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running", "cancelling")

    # ── internals ──
    def _delay(self) -> float:
        return max(get_settings().crawl_site_delay_seconds, self.robots_delay)

    def _throttle(self, url: str) -> bool:
        """Wait for this host's next slot. Returns False if cancelled meanwhile."""
        host = urlparse(url).hostname or ""
        with self._lock:
            now = time.monotonic()
            at = max(now, self._next_at.get(host, 0.0))
            self._next_at[host] = at + self._delay()
        return not self._cancel.wait(max(0.0, at - now)) if at > now else not self._cancel.is_set()

    def _in_scope(self, url: str) -> bool:
        return in_scope(url, self.host, self.include_subdomains, self.path_prefix)

    def _hop_check(self, url: str) -> str | None:
        if self._in_scope(url):
            return None
        # Report private targets as such (a blocked SSRF attempt), everything else as off-scope.
        if ingestion.validate_public_url(url) == ingestion.PRIVATE_URL_MESSAGE:
            return ingestion.PRIVATE_URL_MESSAGE
        return "Redirected outside the crawl scope."

    def _robots_ok(self, url: str) -> bool:
        if not get_settings().crawl_respect_robots:
            return True
        return ingestion.robots_allowed(url)

    def _enqueue(self, frontier: deque, url: str, depth: int, parent: str | None) -> None:
        with self._lock:
            if url in self.seen or not self._in_scope(url) or _SKIP_EXT.search(urlparse(url).path):
                return
            self.seen.add(url)
            self.counts["found"] += 1
            if len(self.pages) >= self.max_pages * 3:
                return
            if not self._robots_ok(url):
                self.pages[url] = {"url": url, "depth": depth, "status": "skipped",
                                   "error": ingestion.ROBOTS_MESSAGE, "parent": parent}
                self.counts["skipped"] += 1
                return
            self.pages[url] = {"url": url, "depth": depth, "status": "queued", "parent": parent}
            frontier.append((url, depth))

    def _set(self, url: str, **fields) -> None:
        with self._lock:
            page = self.pages.setdefault(url, {"url": url})
            old = page.get("status")
            page.update(fields)
            new = page.get("status")
            if old != new:
                if old in ("failed", "skipped", "ingested") and self.counts.get(old, 0) > 0:
                    self.counts[old] -= 1
                if new in ("failed", "skipped", "ingested"):
                    self.counts[new] += 1

    def _discover_sitemaps(self, frontier: deque) -> None:
        settings = get_settings()
        origin = f"{urlparse(self.start_url).scheme}://{urlparse(self.start_url).netloc}"
        queue = []
        try:
            queue += list(ingestion.robots_parser(self.start_url).site_maps() or [])
        except Exception:  # robots problems never stop a crawl
            pass
        queue.append(origin + "/sitemap.xml")
        done, budget = set(), settings.crawl_site_max_sitemap_urls
        while queue and len(done) < 25 and not self._cancel.is_set():
            sm = canonicalize(queue.pop(0))
            if not sm or sm in done or ingestion.validate_public_url(sm):
                continue
            done.add(sm)
            if not self._throttle(sm):
                return
            r = ingestion.fetch_url(sm, allowed_types=SITEMAP_TYPES, max_bytes=10 * 1024 * 1024,
                                    accept="application/xml,text/xml;q=0.9,*/*;q=0.1")
            if r["body"] is None:
                continue
            locs, is_index = parse_sitemap(r["body"])
            if is_index:
                queue += locs[:50]
                continue
            for loc in locs[:budget]:
                u = canonicalize(loc)
                if u and self._in_scope(u):
                    before = self.counts["found"]
                    self._enqueue(frontier, u, 1, "sitemap.xml")
                    if self.counts["found"] > before:
                        self.sitemap_urls += 1
                        budget -= 1
            if budget <= 0:
                break

    def _fetch_page(self, url: str, depth: int, frontier: deque) -> None:
        from knowledge_forge import js_render

        settings = get_settings()
        self._set(url, status="fetching", error="")
        problem = ingestion.validate_public_url(url)
        if problem:
            self._set(url, status="failed", error=problem)
            return
        if not self._throttle(url):
            self._set(url, status="queued")
            return
        r = ingestion.fetch_url(url, allowed_types=PAGE_TYPES + PDF_TYPES,
                                max_bytes=max(settings.crawl_max_bytes, settings.max_upload_bytes),
                                accept="text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.1",
                                hop_check=self._hop_check)
        with self._lock:
            self.counts["fetched"] += 1
        final = canonicalize(r["url"]) or url
        if r["body"] is None:
            status = "skipped" if "outside the crawl scope" in r["error"] or "Unsupported" in r["error"] else "failed"
            self._set(url, status=status, error=r["error"], final_url=final)
            return
        if final != url:
            with self._lock:
                if final in self.seen and final in self.pages and self.pages[final].get("status") != "queued":
                    self._set(url, status="skipped", error="Duplicate (redirects to an already crawled page).",
                              final_url=final)
                    return
                self.seen.add(final)
            if settings.crawl_respect_robots and not ingestion.robots_allowed(final):
                self._set(url, status="skipped", error=ingestion.ROBOTS_MESSAGE, final_url=final)
                return

        title, text, kind = "", "", "html"
        if r["ctype"] in PDF_TYPES or r["body"][:5] == b"%PDF-":
            kind = "pdf"
            try:
                text = ingestion.ingest_pdf_bytes(r["body"]) or ""
            except IngestionError as exc:
                self._set(url, status="failed", error=str(exc), kind=kind, final_url=final)
                return
            title = unquote(posixpath.basename(urlparse(final).path)) or final
        elif r["ctype"] == "text/plain":
            text = r["body"].decode(r["encoding"] or "utf-8", errors="replace")
            title = posixpath.basename(urlparse(final).path) or final
        else:
            info = parse_html(r["body"], final)
            mode = self.render_js
            if mode == "always" or (mode == "auto" and js_render.looks_like_spa(r["body"], info["text"])):
                try:
                    rendered, _ = js_render.render_page(final)
                    rinfo = parse_html(rendered, final)
                    if len(rinfo["text"]) >= len(info["text"]):
                        info = rinfo
                except IngestionError as exc:
                    if len(info["text"].strip()) < settings.crawl_min_chars:
                        self._set(url, status="failed", error=str(exc), final_url=final)
                        return
            title, text = info["title"], info["text"]
            if info["canonical"] and info["canonical"] != final and self._in_scope(info["canonical"]):
                with self._lock:
                    dup = info["canonical"] in self.pages and self.pages[info["canonical"]].get("status") in (
                        "ingested", "fetching")
                    self.seen.add(info["canonical"])
                if dup:
                    self._set(url, status="skipped", error="Duplicate of its canonical URL.", final_url=final)
                    return
            if not info["nofollow"] and depth < self.max_depth:
                for link in info["links"]:
                    self._enqueue(frontier, link, depth + 1, url)
            if info["noindex"]:
                self._set(url, status="skipped", error="Page asks not to be indexed (meta robots noindex).",
                          title=title, final_url=final)
                return

        if len(text.strip()) < settings.crawl_min_chars:
            self._set(url, status="skipped", error="Too little readable text.", title=title, kind=kind,
                      final_url=final)
            return
        digest = hashlib.sha256(normalize_text(text).encode()).hexdigest()
        with self._lock:
            if digest in self.hashes:
                self._set(url, status="skipped", error="Duplicate content.", title=title, kind=kind,
                          final_url=final)
                return
            self.hashes.add(digest)
        meta = {"group": self.group, "site": _bare(urlparse(final).hostname or self.host), "title": title,
                "crawl_id": self.id, "url": final, "depth": depth, "kind": kind}
        try:
            res = store_text(self.kb, text, final, "pdf" if kind == "pdf" else "web", meta)
        except Exception as exc:  # embedding/quota errors are per page
            logger.warning("Crawl ingest failed: %s", type(exc).__name__)
            self._set(url, status="failed", error=str(exc)[:300] or type(exc).__name__, title=title, kind=kind,
                      final_url=final)
            return
        with self._lock:
            self.counts["chunks"] += res["total"]
        self._set(url, status="ingested", title=title, kind=kind, final_url=final, chunks=res["total"],
                  added=res["added"])

    def run(self, seeds: list[tuple[str, int]] | None = None) -> None:
        settings = get_settings()
        with self._lock:
            self.status = "running"
            self.finished = None
        frontier: deque = deque()
        try:
            if seeds is None:
                if settings.crawl_respect_robots:
                    try:
                        delay = ingestion.robots_parser(self.start_url).crawl_delay(settings.crawl_user_agent)
                        self.robots_delay = min(float(delay or 0), 30.0)
                    except Exception:
                        self.robots_delay = 0.0
                self._enqueue(frontier, self.start_url, 0, None)
                if not frontier and self.pages.get(self.start_url, {}).get("status") == "skipped":
                    self.error = ingestion.ROBOTS_MESSAGE
                elif self.use_sitemap and self.max_depth >= 1:
                    self._discover_sitemaps(frontier)
            else:
                for url, depth in seeds:
                    self._set(url, status="queued", error="")
                    frontier.append((url, depth))
            with ThreadPoolExecutor(max_workers=settings.crawl_site_concurrency) as pool:
                inflight = set()
                while (frontier or inflight) and not self._cancel.is_set():
                    while frontier and len(inflight) < settings.crawl_site_concurrency:
                        if seeds is None and self._dispatched >= self.max_pages:
                            frontier.clear()
                            break
                        url, depth = frontier.popleft()
                        self._dispatched += 1
                        inflight.add(pool.submit(self._safe_fetch, url, depth, frontier))
                    if inflight:
                        _done, inflight = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
                if self._cancel.is_set():
                    for f in inflight:
                        f.cancel()
            with self._lock:
                for page in self.pages.values():
                    if page.get("status") in ("queued", "fetching"):
                        page["status"] = "not_fetched"
                self.status = "cancelled" if self._cancel.is_set() else "done"
        except Exception as exc:
            logger.exception("Crawl job failed")
            with self._lock:
                self.status, self.error = "failed", str(exc)[:300]
        finally:
            self.finished = time.time()

    def _safe_fetch(self, url: str, depth: int, frontier: deque) -> None:
        try:
            self._fetch_page(url, depth, frontier)
        except Exception as exc:
            logger.warning("Crawl page error: %s", type(exc).__name__)
            self._set(url, status="failed", error=str(exc)[:300] or type(exc).__name__)

    def retry(self, urls: list[str] | None = None) -> int:
        with self._lock:
            if self.active:
                raise ValidationError("Wait for the crawl to finish before retrying pages.")
            wanted = set(urls or [])
            seeds = [
                (u, p.get("depth", 0)) for u, p in self.pages.items()
                if (u in wanted if wanted else p.get("status") in ("failed", "not_fetched"))
            ]
            if not seeds:
                return 0
            self._cancel.clear()
            self.status = "queued"
        self.start(seeds)
        return len(seeds)

    def start(self, seeds=None) -> None:
        self._thread = threading.Thread(target=self.run, args=(seeds,), name=f"crawl-{self.id}", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)


class CrawlManager:
    def __init__(self) -> None:
        self._jobs: dict[str, CrawlJob] = {}
        self._lock = threading.Lock()

    def start(self, kb, url: str, **opts) -> CrawlJob:
        settings = get_settings()
        url = canonicalize(ingestion.normalize_url(url))
        if not url:
            raise ValidationError("Enter a valid http(s) URL to crawl.")
        problem = ingestion.validate_public_url(url)
        if problem:
            raise ValidationError(problem)
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.active)
            if running >= settings.crawl_site_max_jobs:
                raise ValidationError("Too many crawls are running. Wait for one to finish or cancel it.")
            job = CrawlJob(kb, url, **opts)
            self._jobs[job.id] = job
            for old in sorted(self._jobs.values(), key=lambda j: j.created)[:-20]:
                if not old.active:
                    self._jobs.pop(old.id, None)
        job.start()
        return job

    def get(self, job_id: str) -> CrawlJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError("Crawl job not found (jobs are kept in memory and lost on restart).")
        return job

    def list(self) -> list[dict]:
        return [j.snapshot(include_pages=False) for j in sorted(self._jobs.values(), key=lambda j: -j.created)]


_manager = CrawlManager()


def get_crawl_manager() -> CrawlManager:
    return _manager
