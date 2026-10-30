"""Multimodal content ingestion helpers."""
from __future__ import annotations

import ipaddress
import logging
import mimetypes
import os
import re
import socket
import tempfile
import threading
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from knowledge_forge import gemini
from knowledge_forge.config import get_settings
from knowledge_forge.errors import ConfigurationError, IngestionError, KnowledgeForgeError, RateLimitError
from knowledge_forge.llm import ensure_gemini_configured, generate_content

logger = logging.getLogger(__name__)

PRIVATE_URL_MESSAGE = "URLs pointing to local or private networks are not allowed."
_BLOCKED_MARKER = "kf-blocked-non-public-address"

_YT_HOST = re.compile(r"^(?:https?://)?(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be|youtube-nocookie\.com)/", re.I)


def is_youtube_url(url: str) -> bool:
    url = (url or "").strip()
    return bool(_YT_HOST.match(url)) and _extract_youtube_id(url) is not None


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if url and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "https://" + url
    return url


def is_blocked_ip(raw: str) -> bool:
    """True for any address that is not globally routable (private, loopback, link-local,
    CGNAT, unspecified, reserved, multicast, ...), including IPv4-mapped IPv6 forms."""
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        return True
    if ip.version == 6:
        mapped = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if mapped is not None and is_blocked_ip(str(mapped)):
            return True
    return not ip.is_global or ip.is_multicast


def validate_public_url(url: str) -> str | None:
    """Return an error message if ``url`` is not a public http(s) URL, else None.

    This is the friendly pre-check; :func:`_guarded_create_connection` re-checks the
    address actually connected to, which closes the DNS-rebinding window.
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        parsed.port  # noqa: B018 - raises ValueError for malformed ports
    except ValueError:
        return "That URL is malformed."
    if parsed.scheme not in ("http", "https") or not host:
        return "Only http(s) URLs are supported."
    if parsed.username or parsed.password:
        return "URLs with embedded credentials are not supported."
    if get_settings().allow_private_urls:
        return None
    if host.lower() in ("localhost", "localhost.localdomain") or host.lower().endswith((".local", ".internal")):
        return PRIVATE_URL_MESSAGE
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return PRIVATE_URL_MESSAGE if is_blocked_ip(host) else None
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return f"Could not resolve host '{host[:100]}'."
    if not infos or any(is_blocked_ip(info[4][0]) for info in infos):
        return PRIVATE_URL_MESSAGE
    return None


def _guarded_create_connection(address, timeout=None, source_address=None, socket_options=None):
    """Like ``urllib3.util.connection.create_connection`` but refuses non-public peers."""
    host, port = address
    host = host.strip("[]")
    allow_private = get_settings().allow_private_urls
    last: OSError | None = None
    for family, socktype, proto, _canon, sockaddr in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
        if not allow_private and is_blocked_ip(sockaddr[0]):
            last = OSError(f"{_BLOCKED_MARKER}: {sockaddr[0]}")
            continue
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            for opt in socket_options or ():
                sock.setsockopt(*opt)
            if timeout is not None and timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last = exc
            if sock is not None:
                sock.close()
    raise last or OSError(f"getaddrinfo returned no addresses for {host}")


class _GuardedHTTPConnection(HTTPConnection):
    def _new_conn(self):
        return _guarded_create_connection(
            (self._dns_host, self.port), self.timeout, self.source_address, self.socket_options
        )


class _GuardedHTTPSConnection(HTTPSConnection):
    def _new_conn(self):
        return _guarded_create_connection(
            (self._dns_host, self.port), self.timeout, self.source_address, self.socket_options
        )


class _GuardedHTTPPool(HTTPConnectionPool):
    ConnectionCls = _GuardedHTTPConnection


class _GuardedHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _GuardedHTTPSConnection


class _GuardedAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {"http": _GuardedHTTPPool, "https": _GuardedHTTPSPool}


def _new_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # ignore *_PROXY env vars: a proxy would bypass the peer check
    adapter = _GuardedAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_session_local = threading.local()


def _http_get(url: str, **kwargs) -> requests.Response:
    """GET through a per-thread session whose sockets refuse non-public peers."""
    session = getattr(_session_local, "session", None)
    if session is None:
        session = _session_local.session = _new_session()
    return session.get(url, **kwargs)


def _http_request(method: str, url: str, **kwargs) -> requests.Response:
    """Any-method request through the same guarded per-thread session (used by the JS renderer)."""
    session = getattr(_session_local, "session", None)
    if session is None:
        session = _session_local.session = _new_session()
    return session.request(method, url, **kwargs)


_HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/xml", "text/xml")


def _read_capped(resp, limit: int) -> bytes:
    declared = resp.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise IngestionError(f"The page is larger than the {limit // 1024} KB crawl limit.")
    buf = bytearray()
    for chunk in resp.iter_content(64 * 1024):
        buf.extend(chunk)
        if len(buf) > limit:
            raise IngestionError(f"The page is larger than the {limit // 1024} KB crawl limit.")
    return bytes(buf)


def _html_to_text(body: bytes | str) -> str:
    soup = BeautifulSoup(body, "html.parser")
    for el in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "svg", "template"]):
        el.decompose()
    main = soup.find("article") or soup.find("main") or soup.body
    return " ".join((main or soup).stripped_strings)


def _fetch_static(url: str) -> tuple[bytes | None, str, str, str]:
    """Fetch ``url`` with redirect re-validation. Returns ``(body, ctype, encoding, error)``."""
    settings = get_settings()
    headers = {
        "User-Agent": settings.crawl_user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.1",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = None
    try:
        for _hop in range(settings.crawl_max_redirects + 1):
            resp = _http_get(url, headers=headers, timeout=settings.crawl_timeout_seconds,
                             allow_redirects=False, stream=True)
            location = resp.headers.get("location") if 300 <= resp.status_code < 400 else None
            if not location:
                break
            resp.close()
            url = urljoin(url, location)
            problem = validate_public_url(url)
            if problem:
                logger.warning("Blocked redirect to a non-public URL")
                return None, "", "", problem
        else:
            return None, "", "", "The page redirected too many times."
        if resp.status_code >= 400:
            return None, "", "", f"The site responded with HTTP {resp.status_code}."
        ctype = (resp.headers.get("content-type") or "text/html").split(";")[0].strip().lower()
        if ctype not in _HTML_TYPES:
            return None, ctype, "", f"Unsupported content type '{ctype[:60]}'. Upload files with the File tab."
        body = _read_capped(resp, settings.crawl_max_bytes)
        return body, ctype, resp.encoding or "utf-8", ""
    except IngestionError as exc:
        return None, "", "", str(exc)
    except requests.exceptions.Timeout:
        return None, "", "", "The site took too long to respond."
    except requests.exceptions.RequestException as exc:
        if _BLOCKED_MARKER in repr(exc):
            return None, "", "", PRIVATE_URL_MESSAGE
        logger.warning("Error crawling URL: %s", type(exc).__name__)
        return None, "", "", "Could not connect to the site."
    finally:
        if resp is not None:
            resp.close()


_ROBOTS_CACHE: dict[str, tuple[float, object]] = {}
_ROBOTS_TTL = 3600.0
_ROBOTS_MAX_BYTES = 512 * 1024
_SITEMAP_RE = re.compile(rb"<(?:\w+:)?(?:urlset|sitemapindex)[\s>]")
ROBOTS_MESSAGE = "The site's robots.txt does not allow crawling this page."


def robots_allowed(url: str) -> bool:
    """RFC 9309 check of ``url`` against its origin's robots.txt (cached for an hour).

    4xx means no rules; 5xx or an unreachable robots.txt is treated as "allow" so a flaky
    robots server does not block the page fetch, which has its own error handling.
    """
    import urllib.robotparser

    settings = get_settings()
    parts = urlparse(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    now = time.monotonic()
    cached = _ROBOTS_CACHE.get(origin)
    if cached is None or now - cached[0] > _ROBOTS_TTL:
        parser = urllib.robotparser.RobotFileParser()
        lines: list[str] = []
        try:
            resp = _http_get(origin + "/robots.txt", headers={"User-Agent": settings.crawl_user_agent},
                             timeout=min(10, settings.crawl_timeout_seconds), stream=True)
            try:
                if resp.status_code < 300:
                    lines = _read_capped(resp, _ROBOTS_MAX_BYTES).decode("utf-8", "replace").splitlines()
            finally:
                resp.close()
        except (requests.RequestException, IngestionError):
            logger.info("robots.txt unavailable; allowing")
        parser.parse(lines)
        if len(_ROBOTS_CACHE) > 512:
            _ROBOTS_CACHE.clear()
        _ROBOTS_CACHE[origin] = (now, parser)
        cached = _ROBOTS_CACHE[origin]
    return cached[1].can_fetch(settings.crawl_user_agent, url)


def resolve_render_mode(requested: str | None) -> str:
    """Combine the server's RENDER_JS setting with a per-request override.

    ``never`` on the server disables rendering entirely; otherwise a request may ask for
    ``auto``, ``always`` or ``never``.
    """
    server = get_settings().render_js
    req = (requested or "").strip().lower()
    if not req:
        return server
    if req not in ("auto", "always", "never"):
        raise IngestionError("render_js must be one of auto, always or never.")
    if server == "never" and req != "never":
        raise IngestionError("JavaScript rendering is disabled on this server (RENDER_JS=never).")
    return req


def crawl_url_detailed(url: str, render_js: str | None = None) -> tuple[str | None, str]:
    """Fetch ``url`` and extract readable text. Returns ``(text, error_reason)``.

    ``render_js`` is ``auto`` / ``always`` / ``never`` (default: the RENDER_JS setting).
    """
    from knowledge_forge import js_render

    settings = get_settings()
    url = normalize_url(url)
    problem = validate_public_url(url)
    if problem:
        return None, problem
    if is_youtube_url(url):
        return None, "That is a YouTube link; add it as a YouTube source."
    try:
        mode = resolve_render_mode(render_js)
    except IngestionError as exc:
        return None, str(exc)
    if settings.crawl_respect_robots and not robots_allowed(url):
        return None, ROBOTS_MESSAGE

    text, reason, html = "", "", b""
    if mode != "always":
        body, ctype, encoding, reason = _fetch_static(url)
        if body is not None and ctype in ("application/xml", "text/xml") and _SITEMAP_RE.search(body[:4096]):
            return None, "This is an XML sitemap, not a page. Add the pages it lists instead."
        if body is not None:
            html = body
            text = body.decode(encoding, errors="replace") if ctype == "text/plain" else _html_to_text(body)
            if ctype == "text/plain":
                mode = "never"
        else:
            # HTTP errors, timeouts and size limits would hit the browser the same way.
            return None, reason

    if mode == "always" or (mode == "auto" and (not text or js_render.looks_like_spa(html, text))):
        try:
            rendered, _final = js_render.render_page(url)
            rendered_text = _html_to_text(rendered)
            if len(rendered_text.strip()) >= len(text.strip()):
                text = rendered_text
        except IngestionError as exc:
            if len(text.strip()) < settings.crawl_min_chars:
                return None, str(exc)
            logger.info("JavaScript rendering failed; using static HTML")

    if len(text.strip()) < settings.crawl_min_chars:
        if reason:
            return None, reason
        hint = ("Turn on 'Render JavaScript' or paste it as Text instead." if js_render.status()["available"]
                else "Paste it as Text instead.")
        return None, f"Could not extract readable text (the page may need JavaScript). {hint}"
    return text, ""


def crawl_url(url: str, render_js: str | None = None) -> str | None:
    return crawl_url_detailed(url, render_js)[0]


def _extract_youtube_id(video_url: str) -> str | None:
    video_url = (video_url or "").strip()
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    return None


def _fetch_youtube_oembed(watch_url: str) -> str | None:
    """Lightweight metadata via oEmbed (avoids scraping watch page / 429 blocks)."""
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": watch_url, "format": "json"},
            headers={"User-Agent": get_settings().crawl_user_agent},
            timeout=get_settings().crawl_timeout_seconds,
        )
        r.raise_for_status()
        data = r.json()
        title = data.get("title", "").strip()
        author = data.get("author_name", "").strip()
        if not title:
            return None
        parts = [f"Title: {title}"]
        if author:
            parts.append(f"Channel: {author}")
        return "\n".join(parts)
    except Exception as exc:
        logger.warning("YouTube oEmbed failed: %s", type(exc).__name__)
        return None


def _youtube_gemini_fallback(video_url: str, page_text: str | None) -> str | None:
    """Last resort: use Gemini on scraped title/description."""
    if not page_text:
        return None
    try:
        from knowledge_forge.prompts import youtube_metadata_prompt

        return generate_content(youtube_metadata_prompt(video_url, page_text[:8000])) or None
    except KnowledgeForgeError as exc:
        logger.error("YouTube metadata fallback failed: %s", str(exc))
        return None


def get_youtube_transcript(video_url: str) -> tuple[str | None, str]:
    """
    Fetch YouTube content. Returns (text, error_detail).
    error_detail is empty string on success.
    """
    vid_id = _extract_youtube_id(video_url)
    if not vid_id:
        return None, "Invalid YouTube URL. Use a link like https://youtube.com/watch?v=VIDEO_ID"

    watch_url = f"https://www.youtube.com/watch?v={vid_id}"
    last_error = ""

    blocked_by_youtube = False

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            IpBlocked,
            NoTranscriptFound,
            RequestBlocked,
            TranscriptsDisabled,
            VideoUnavailable,
        )

        api = YouTubeTranscriptApi()
        language_sets = [list(get_settings().youtube_languages or ("en",))]
        for langs in language_sets:
            try:
                fetched = api.fetch(vid_id, languages=langs)
                text = " ".join(s.text for s in fetched.snippets)
                if text.strip():
                    return text, ""
            except NoTranscriptFound as exc:
                last_error = str(exc)
                continue
            except (RequestBlocked, IpBlocked):
                blocked_by_youtube = True
                last_error = "YouTube blocked requests from this server IP (common on cloud hosts)."
                break
            except TranscriptsDisabled:
                last_error = "Transcripts are disabled for this video."
                break
            except VideoUnavailable:
                return None, "Video is unavailable or private."

        if last_error != "Transcripts are disabled for this video.":
            try:
                transcript_list = api.list(vid_id)
                for transcript in transcript_list:
                    try:
                        fetched = transcript.fetch()
                        text = " ".join(s.text for s in fetched.snippets)
                        if text.strip():
                            return text, ""
                    except Exception as exc:
                        last_error = str(exc)
            except VideoUnavailable:
                return None, "Video is unavailable or private."
            except Exception as exc:
                last_error = str(exc)

    except ImportError:
        last_error = "youtube-transcript-api is not installed. Run: pip install -r requirements.txt"
    except Exception as exc:
        last_error = str(exc)
        logger.error("YouTube transcript error for %s: %s", vid_id, type(exc).__name__)

    # Opt-in fallback: notes generated from title/channel only. Off by default because the
    # result is the model's guess at the content, not the video's actual content.
    meta = _fetch_youtube_oembed(watch_url) if get_settings().youtube_metadata_fallback else None
    if meta:
        enriched = f"{meta}\nVideo URL: {watch_url}\nVideo ID: {vid_id}"
        gemini_text = _youtube_gemini_fallback(watch_url, enriched)
        if gemini_text:
            logger.info("YouTube ingest used Gemini+oEmbed fallback for %s", vid_id)
            return gemini_text, ""

    if blocked_by_youtube:
        return None, (
            "YouTube blocked transcript requests from this server (Render/cloud IP). "
            "Use the YouTube tab (not Web), ensure captions are enabled, or paste the transcript under Text."
        )

    hint = (
        "Could not fetch captions. Use the YouTube tab, ensure subtitles are enabled, "
        "or paste the transcript as Text."
    )
    if last_error:
        hint = f"{hint} ({last_error})"
    return None, hint


SOURCE_TYPES = ("web", "youtube", "text")


def ingest_text_source(source: str, source_type: str, render_js: str | None = None) -> tuple[str | None, str, str]:
    """
    Ingest URL or raw text. Returns (text, error_detail, effective_source_type).
    Auto-detects YouTube URLs even when submitted as 'web'.
    """
    source = (source or "").strip()
    stype = (source_type or "").strip().lower()

    if stype != "text" and is_youtube_url(source):
        text, detail = get_youtube_transcript(source)
        return text, detail, "youtube"

    if stype == "youtube":
        text, detail = get_youtube_transcript(source)
        return text, detail, "youtube"

    if stype == "text":
        return source, "", "text"

    if stype == "web":
        source = normalize_url(source)
        problem = validate_public_url(source)
        if problem:
            return None, problem, "web"
        text, reason = crawl_url_detailed(source, render_js)
        if not text:
            return None, reason or "Could not extract readable text. Paste the content as Text instead.", "web"
        return text, "", "web"

    return None, f"Unknown source type: {stype}", stype


PDF_DAMAGED_MESSAGE = "This PDF could not be read. It may be damaged or truncated."
PDF_NO_TEXT_MESSAGE = ("This PDF has no text layer (it looks scanned or image-only) and OCR with Gemini "
                       "did not return any text.")


def _ocr_pdf(data: bytes) -> str | None:
    try:
        return ingest_pdf_via_gemini_bytes(data)
    except ConfigurationError:
        logger.info("PDF OCR skipped: Gemini is not configured")
        return None


def ingest_pdf_bytes(data: bytes) -> str | None:
    try:
        import pymupdf
    except ImportError:
        return ingest_pdf_via_gemini_bytes(data)
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise IngestionError("This PDF is password protected. Remove the password and try again.")
            texts = [page.get_text("text") for page in doc]
            repaired = bool(getattr(doc, "is_repaired", False))
    except IngestionError:
        raise
    except Exception as exc:
        logger.error("PDF extraction error: %s", type(exc).__name__)
        text = _ocr_pdf(data)
        if not text:
            raise IngestionError(PDF_DAMAGED_MESSAGE) from None
        return text
    raw = "\n".join(texts).strip()
    # Very little text usually means a scanned PDF: try Gemini OCR, but keep a short real text layer
    # (e.g. a one-line PDF) if OCR is unavailable or returns less.
    if len(raw) < get_settings().pdf_ocr_min_chars:
        ocr = _ocr_pdf(data)
        if ocr and len(ocr.strip()) > len(raw):
            return ocr
        if not raw:
            raise IngestionError(PDF_DAMAGED_MESSAGE if repaired else PDF_NO_TEXT_MESSAGE)
    return raw


def ingest_pdf_via_gemini_bytes(data: bytes) -> str | None:
    return ingest_file_via_gemini(
        data,
        "application/pdf",
        "Extract and return all the text content from this PDF document. Preserve structure.",
    )


def ingest_file_via_gemini(data: bytes, mime_type: str, prompt: str) -> str | None:
    ext_map = {
        "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-m4a": ".m4a",
        "audio/ogg": ".ogg", "video/mp4": ".mp4", "video/webm": ".webm",
        "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "application/pdf": ".pdf",
    }
    suffix = ext_map.get(mime_type, ".bin")
    tmp_path = None
    uploaded = None
    try:
        ensure_gemini_configured()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        uploaded = gemini.upload_file(tmp_path, mime_type)
        for _ in range(get_settings().gemini_file_max_polls):
            f = gemini.get_file(uploaded.name)
            state = getattr(f.state, "name", None) or str(f.state)
            state = state.rsplit(".", 1)[-1]
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini file processing failed.")
            time.sleep(get_settings().gemini_file_poll_seconds)
        else:
            raise TimeoutError("Gemini file processing timed out.")

        return gemini.generate([uploaded, prompt]) or None
    except (ConfigurationError, RateLimitError):
        raise
    except KnowledgeForgeError as exc:
        logger.error("Gemini file ingestion error (%s): %s", mime_type, str(exc))
        return None
    except Exception as exc:
        logger.error("Gemini file ingestion error (%s): %s", mime_type, type(exc).__name__)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if uploaded:
            try:
                gemini.delete_file(uploaded.name)
            except Exception:
                logger.debug("Could not delete uploaded Gemini file", exc_info=True)


def detect_file_type(filename: str, mime_type: str) -> tuple[str, str] | None:
    """Return (source_type, effective_mime) or None if unsupported."""
    lower = filename.lower()
    if mime_type == "application/pdf" or lower.endswith(".pdf"):
        return "pdf", mime_type or "application/pdf"
    if mime_type.startswith("audio/") or lower.endswith((".mp3", ".wav", ".m4a", ".ogg")):
        return "audio", mime_type or mimetypes.guess_type(filename)[0] or "audio/mpeg"
    if mime_type.startswith("video/") or lower.endswith((".mp4", ".webm", ".mov")):
        return "video", mime_type or mimetypes.guess_type(filename)[0] or "video/mp4"
    if mime_type.startswith("image/") or lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return "image", mime_type or mimetypes.guess_type(filename)[0] or "image/jpeg"
    if lower.endswith((".txt", ".md", ".markdown")) or mime_type in ("text/plain", "text/markdown"):
        return "text", "text/plain"
    return None


def _magic_kind(head: bytes) -> str | None:
    """Identify a file family from its leading bytes."""
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")):
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio"
    if head.startswith((b"ID3", b"OggS", b"fLaC")) or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        return "audio" if brand in (b"M4A ", b"M4B ") else "video"
    if head.startswith(b"\x1aE\xdf\xa3"):
        return "video"
    return None


def sniff_file_type(filename: str, mime_type: str, head: bytes) -> tuple[str, str] | None:
    """Like :func:`detect_file_type` but verifies the content's magic bytes.

    Returns None when the declared type is unsupported or the bytes do not match it
    (e.g. an executable renamed to ``.pdf``). Text files must be NUL-free UTF-8-ish text.
    """
    declared = detect_file_type(filename, mime_type or "")
    if not declared:
        return None
    kind, mime = declared
    if kind == "text":
        if b"\x00" in head:
            return None
        return declared
    actual = _magic_kind(head)
    if actual != kind:
        return None
    if kind == "audio" and head[4:8] == b"ftyp":
        mime = "audio/x-m4a"
    return kind, mime


def safe_filename(name: str, max_len: int = 120) -> str:
    """Strip directories/control characters from an uploaded filename for display."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip().strip(".") or "upload"
    return base[:max_len]
