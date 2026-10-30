"""Optional headless-browser rendering for JavaScript-heavy pages (SPAs, lazy content).

Install with ``pip install -r requirements-js.txt && playwright install chromium`` (or the
``js`` extra). Every request the page makes, including the top-level navigation and each
redirect hop, is intercepted and fetched by KnowledgeForge's own SSRF-guarded HTTP client
(which re-checks the connected IP, closing DNS rebinding). Chromium itself never opens a
connection. Images, media and fonts are dropped, WebSockets are refused, service workers
are blocked, and the number of requests, bytes, time and concurrent renders are capped.
"""
from __future__ import annotations

import importlib.util
import logging
import re
import threading
import time
from html import escape as html_escape
from urllib.parse import urljoin

import requests

from knowledge_forge.config import get_settings
from knowledge_forge.errors import IngestionError

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "JavaScript rendering needs Playwright and Chromium. Install them with "
    "`pip install -r requirements-js.txt && playwright install chromium`, "
    "or build the Docker image with `--build-arg INSTALL_JS=true`."
)
_SKIP_TYPES = {"image", "media", "font"}
_HOP_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
_SPA_ROOT = re.compile(
    r"<div[^>]+id=[\"'](?:root|app|__next|__nuxt|svelte|q-app|main-app)[\"'][^>]*>\s*</div>", re.I
)
_NOSCRIPT_JS = re.compile(r"<noscript[^>]*>[^<]{0,300}(?:enable|requires?|need)[^<]{0,40}javascript", re.I)

_sem_lock = threading.Lock()
_sem: threading.BoundedSemaphore | None = None
_sem_size = 0


def playwright_installed() -> bool:
    try:
        return importlib.util.find_spec("playwright.sync_api") is not None
    except (ImportError, ValueError):
        return False


def status() -> dict:
    s = get_settings()
    return {"mode": s.render_js, "available": s.render_js != "never" and playwright_installed()}


def looks_like_spa(html: str | bytes, text: str) -> bool:
    """Heuristic for RENDER_JS=auto: little readable text, or an empty SPA mount point."""
    if isinstance(html, bytes):
        html = html[:200_000].decode("utf-8", errors="replace")
    if len(text.strip()) < get_settings().render_js_min_chars:
        return True
    return bool(_SPA_ROOT.search(html) or _NOSCRIPT_JS.search(html)) and len(text.strip()) < 2000


def _semaphore() -> threading.BoundedSemaphore:
    global _sem, _sem_size
    size = get_settings().render_js_max_concurrency
    with _sem_lock:
        if _sem is None or _sem_size != size:
            _sem, _sem_size = threading.BoundedSemaphore(size), size
        return _sem


class _Blocked(Exception):
    pass


def _check_url(url: str) -> str | None:
    from knowledge_forge.ingestion import validate_public_url

    return validate_public_url(url)


def _guarded_fetch(method: str, url: str, headers: dict, body: bytes | None, timeout: float, cap: int):
    from knowledge_forge.ingestion import _http_request

    resp = _http_request(method, url, headers=headers, data=body, timeout=timeout,
                     allow_redirects=False, stream=True)
    try:
        chunks, size = [], 0
        for chunk in resp.iter_content(64 * 1024):
            size += len(chunk)
            if size > cap:
                raise IngestionError("A resource on the page exceeded the size limit.")
            chunks.append(chunk)
        return resp.status_code, dict(resp.headers), b"".join(chunks)
    finally:
        resp.close()


def render_page(url: str) -> tuple[str, str]:
    """Render ``url`` in headless Chromium. Returns ``(html, final_url)``.

    Raises IngestionError with a user-facing message on any failure.
    """
    settings = get_settings()
    if settings.render_js == "never":
        raise IngestionError("JavaScript rendering is disabled on this server (RENDER_JS=never).")
    problem = _check_url(url)
    if problem:
        raise IngestionError(problem)
    try:
        from playwright.sync_api import Error as PWError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise IngestionError(INSTALL_HINT) from exc

    sem = _semaphore()
    if not sem.acquire(timeout=settings.render_js_timeout_seconds):
        raise IngestionError("The page renderer is busy. Please try again in a moment.")
    deadline = time.monotonic() + settings.render_js_timeout_seconds
    budget = {"requests": 0, "bytes": 0}
    blocked: list[str] = []
    nav_blocked: list[str] = []

    def handle(route, request):
        req_url = request.url
        if request.resource_type in _SKIP_TYPES:
            return route.abort("blockedbyclient")
        budget["requests"] += 1
        remaining = deadline - time.monotonic()
        if budget["requests"] > settings.render_js_max_requests or remaining <= 0:
            return route.abort("blockedbyclient")
        reason = _check_url(req_url)
        if reason:
            blocked.append(req_url)
            if request.is_navigation_request():
                nav_blocked.append(req_url)
            logger.warning("Blocked browser request to a non-public or unsupported URL")
            return route.abort("blockedbyclient")
        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        headers["User-Agent"] = settings.crawl_user_agent
        method, body_in, target = request.method, request.post_data_buffer, req_url
        try:
            # Redirects are followed here, hop by hop, so every hop is re-validated; Chromium
            # does not re-intercept redirects of fulfilled responses, so it must never see a 3xx.
            for _hop in range(settings.crawl_max_redirects + 1):
                status_code, resp_headers, body = _guarded_fetch(
                    method, target, headers, body_in,
                    timeout=max(1.0, min(deadline - time.monotonic(), settings.crawl_timeout_seconds)),
                    cap=max(0, settings.crawl_max_bytes - budget["bytes"]))
                budget["bytes"] += len(body)
                location = {k.lower(): v for k, v in resp_headers.items()}.get("location")
                if not (300 <= status_code < 400 and location):
                    break
                target = urljoin(target, location)
                if status_code in (301, 302, 303) and method != "HEAD":
                    method, body_in = "GET", None
                if _check_url(target):
                    raise _Blocked(target)
            else:
                return route.abort("failed")
        except _Blocked:
            blocked.append(target)
            if request.is_navigation_request():
                nav_blocked.append(target)
            logger.warning("Blocked browser redirect to a non-public URL")
            return route.abort("blockedbyclient")
        except (requests.RequestException, IngestionError, OSError) as exc:
            if "kf-blocked" in repr(exc):  # connect-time guard caught a rebinding attempt
                blocked.append(target)
                if request.is_navigation_request():
                    nav_blocked.append(target)
            return route.abort("failed")
        clean = {k: v for k, v in resp_headers.items() if k.lower() not in _HOP_HEADERS}
        if target != req_url and "html" in clean.get("Content-Type", clean.get("content-type", "")):
            body = re.sub(rb"(<head[^>]*>)", lambda m: m.group(1) + b'<base href="' + html_escape(target).encode()
                          + b'">', body, count=1, flags=re.I)
        return route.fulfill(status=status_code, headers=clean, body=body)

    try:
        with sync_playwright() as pw:
            launch = {"headless": True, "args": [
                "--disable-dev-shm-usage", "--no-first-run",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                "--disable-features=WebRtcHideLocalIpsWithMdns",
            ]}
            if settings.render_js_browser_path:
                launch["executable_path"] = settings.render_js_browser_path
            try:
                browser = pw.chromium.launch(**launch)
            except PWError as exc:
                logger.warning("Chromium failed to launch: %s", str(exc).splitlines()[0][:200])
                raise IngestionError(INSTALL_HINT) from exc
            try:
                context = browser.new_context(
                    user_agent=settings.crawl_user_agent, service_workers="block",
                    java_script_enabled=True, accept_downloads=False,
                )
                context.route("**/*", handle)
                if hasattr(context, "route_web_socket"):
                    context.route_web_socket("**/*", lambda ws: ws.close())
                page = context.new_page()
                timeout_ms = int(max(1.0, deadline - time.monotonic()) * 1000)
                page.set_default_timeout(timeout_ms)
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                except PWError as exc:
                    if nav_blocked:
                        raise IngestionError("The page tried to navigate to a local or private address.") from exc
                    if "Timeout" in type(exc).__name__ or "Timeout" in str(exc)[:80]:
                        raise IngestionError("The page took too long to render.") from exc
                    raise IngestionError("The headless browser could not load the page.") from exc
                if response is not None and response.status >= 400:
                    raise IngestionError(f"The site responded with HTTP {response.status}.")
                settle = min(settings.render_js_settle_ms, int(max(0, deadline - time.monotonic()) * 1000))
                if settle:
                    try:
                        page.wait_for_load_state("networkidle", timeout=settle)
                    except PWError:
                        pass  # long-polling pages never go idle; use what has rendered
                if nav_blocked:
                    raise IngestionError("The page tried to navigate to a local or private address.")
                html = page.content()
                final_url = page.url
                if blocked:
                    logger.info("Rendered page with %d blocked subrequest(s)", len(blocked))
            finally:
                browser.close()
    except IngestionError:
        raise
    except Exception as exc:  # playwright driver crash, etc.
        logger.warning("JavaScript rendering failed: %s", type(exc).__name__)
        raise IngestionError("The headless browser failed while rendering the page.") from exc
    finally:
        sem.release()
    if len(html.encode("utf-8", errors="ignore")) > settings.crawl_max_bytes:
        html = html[: settings.crawl_max_bytes]
    return html, final_url
