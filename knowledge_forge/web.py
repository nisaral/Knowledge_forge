"""Flask application factory and JSON API for KnowledgeForge."""
from __future__ import annotations

import base64
import functools
import io
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque

import numpy as np
from flask import Flask, Response, g, jsonify, render_template, request, stream_with_context

from knowledge_forge import __version__, js_render, studio
from knowledge_forge import create as create_mod
from knowledge_forge import graph as graph_mod
from knowledge_forge.cache import redis_health_check
from knowledge_forge.chunking import content_hash, normalize_text
from knowledge_forge.config import flask_secret, get_settings
from knowledge_forge.crawler import get_crawl_manager, store_text
from knowledge_forge.errors import (
    MISSING_KEY_MESSAGE,
    EmptyKnowledgeBase,
    IngestionError,
    KnowledgeForgeError,
    LLMError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from knowledge_forge.ingestion import (
    SOURCE_TYPES,
    ingest_file_via_gemini,
    ingest_pdf_bytes,
    ingest_text_source,
    resolve_render_mode,
    safe_filename,
    sniff_file_type,
)
from knowledge_forge.knowledge_base import KnowledgeBase, get_knowledge_base
from knowledge_forge.llm import (
    clean_label,
    ensure_gemini_configured,
    generate_content,
    normalize_mermaid,
    parse_json_output,
    stream_content,
    validate_flashcards,
    validate_quiz,
    validate_storyboard,
)
from knowledge_forge.logging_setup import configure_logging
from knowledge_forge.prompts import (
    FILE_PROMPTS,
    JSON_TOOLS,
    SYSTEM_INSTRUCTION,
    TOOL_ALIASES,
    TOOLS,
    answer_prompt,
    chat_prompt,
    cited_numbers,
    cluster_label_prompt,
    entities_prompt,
    repair_json_prompt,
    summary_prompt,
    tool_prompt,
)
from knowledge_forge.search import search as search_kb

logger = logging.getLogger("knowledgeforge")

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in your knowledge base. Add a document, web page or "
    "video on this topic first, or try rephrasing the question."
)

_VALIDATORS = {
    "flashcards": (validate_flashcards, '[{"question": str, "answer": str}]'),
    "storyboard": (validate_storyboard, '[{"title": str, "scene": str}]'),
    "quiz": (validate_quiz, '[{"question": str, "options": [4 x str], "answer": int, "explanation": str}]'),
}


# ─── Helpers ────────────────────────────────────────────────────────────────
def _error(message: str, status: int):
    return jsonify({"success": False, "message": message}), status


def _json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValidationError("Invalid or missing JSON body.")
    return data


def _opt_str(data: dict, key: str, max_len: int | None = None) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"'{key}' must be a string.")
    value = value.strip() or None
    if value and max_len and len(value) > max_len:
        raise ValidationError(f"'{key}' is too long (max {max_len} characters).")
    return value


def _required_str(data: dict, key: str, label: str, max_len: int | None = None) -> str:
    value = _opt_str(data, key)
    if not value:
        raise ValidationError(f"{label} is required.")
    if max_len and len(value) > max_len:
        raise ValidationError(f"{label} is too long (max {max_len} characters).")
    return value


def _citations(hits: list[dict], answer: str | None = None) -> list[dict]:
    """Numbered citations; ``cited`` marks the ones the answer actually references."""
    limit = get_settings().citation_snippet_chars
    used = cited_numbers(answer) if answer is not None else set()
    return [
        {
            "n": i,
            "source": h.get("source", ""),
            "source_type": h.get("source_type", "text"),
            "snippet": (h.get("text") or "")[:limit],
            "score": h.get("score"),
            "cited": i in used,
        }
        for i, h in enumerate(hits, start=1)
    ]


def _clean_history(raw, turns: int) -> list[dict]:
    """Validate client-supplied chat history and keep the last ``turns`` exchanges."""
    if not isinstance(raw, list) or turns <= 0:
        return []
    limit = get_settings().history_message_chars
    cleaned = []
    for item in raw[-turns * 2:]:
        if not isinstance(item, dict):
            continue
        role = "user" if item.get("role") == "user" else "assistant"
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()[:limit]})
    return cleaned


class RateLimiter:
    """Per-client sliding-window limiter for the expensive (LLM-backed) endpoints.

    In-memory, so the limit applies per worker process; put a shared limiter (e.g. at the
    reverse proxy) in front of multi-worker deployments that need a hard global cap.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if self.per_minute <= 0:
            return
        now = time.monotonic()
        with self._lock:
            window = self._hits[key]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                raise RateLimitError("Too many requests. Please wait a minute and try again.")
            window.append(now)
            if len(self._hits) > 10_000:  # drop idle clients so memory stays bounded
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] > 60]:
                    del self._hits[k]


def api_endpoint(fn=None, *, limited: bool = False):
    """Translate domain errors into consistent JSON responses (optionally rate limited)."""

    def decorate(view):
        @functools.wraps(view)
        def wrapper(*args, **kwargs):
            try:
                if limited:
                    from flask import current_app

                    current_app.extensions["kf_rate_limiter"].check(request.remote_addr or "unknown")
                return view(*args, **kwargs)
            except KnowledgeForgeError as exc:
                if exc.status >= 500:
                    logger.warning("%s failed: %s", view.__name__, exc)
                return _error(str(exc), exc.status)
            except Exception:
                logger.exception("%s failed unexpectedly", view.__name__)
                return _error("Something went wrong on the server. Please try again.", 500)

        return wrapper

    return decorate(fn) if fn is not None else decorate


def _scope(kb: KnowledgeBase, data: dict) -> str | None:
    """Return the requested source filter, or 404 if that source does not exist."""
    source = _opt_str(data, "source", get_settings().source_name_max_chars)
    if source and not kb.source_exists(source):
        raise NotFoundError("That source is no longer in your knowledge base. Pick another or clear the focus.")
    return source


def _note_label(text: str, title: str | None) -> str:
    if title:
        return title
    preview = normalize_text(text)[:48].strip()
    return f"Note: {preview} · {content_hash(normalize_text(text))[:6]}"


def _ingest(kb: KnowledgeBase, text: str | None, source: str, source_type: str, metadata=None):
    settings = get_settings()
    res = store_text(kb, text or "", source, source_type, metadata)
    added, removed, truncated = res["added"], res["removed"], res["truncated"]
    if added and removed:
        message = f"Updated {source_type} source: {added} new chunk{'s' if added != 1 else ''}, {removed} removed."
    elif added:
        message = f"Added {added} chunk{'s' if added != 1 else ''} from {source_type}."
    elif removed:
        message = f"Updated {source_type} source: removed {removed} outdated chunk{'s' if removed != 1 else ''}."
    else:
        message = "This content is already in your knowledge base; nothing new to add."
    if truncated:
        message += f" (Content was truncated to {settings.max_text_chars:,} characters.)"
    return jsonify({
        "success": True,
        "message": message,
        "chunks_added": added,
        "chunks_removed": removed,
        "chunks_total": res["total"],
        "source": source,
        "source_type": source_type,
    })


def _balanced_text(chunks: list[dict], budget: int) -> str:
    """Take chunks round-robin across sources (in document order) up to ``budget`` chars."""
    by_source: dict[str, list[str]] = {}
    for c in chunks:
        by_source.setdefault(c["source"], []).append(c["text"])
    queues = [deque(v) for v in by_source.values()]
    parts: list[str] = []
    used = 0
    while queues and used < budget:
        for q in list(queues):
            if not q:
                queues.remove(q)
                continue
            piece = q.popleft()[: budget - used]
            parts.append(piece)
            used += len(piece) + 1
            if used >= budget:
                break
    return "\n".join(parts)


def _structured(tool: str, prompt: str) -> list[dict]:
    """Generate JSON for ``tool`` and validate it, with one repair round-trip."""
    validate, shape = _VALIDATORS[tool]
    raw = generate_content(prompt, system=SYSTEM_INSTRUCTION, json_mode=True)
    try:
        items = validate(parse_json_output(raw))
    except LLMError:
        items = []
    if not items:
        logger.info("Structured %s output invalid; asking the model to repair it", tool)
        repaired = generate_content(repair_json_prompt(raw, shape), json_mode=True)
        try:
            items = validate(parse_json_output(repaired))
        except LLMError:
            items = []
    if not items:
        raise LLMError("The model returned an unexpected format. Please try again.")
    return items


def _opt_int(data: dict, key: str, default: int, lo: int, hi: int) -> int:
    value = data.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValidationError(f"'{key}' must be a number.")
    try:
        value = int(value)
    except ValueError as exc:
        raise ValidationError(f"'{key}' must be a number.") from exc
    if not lo <= value <= hi:
        raise ValidationError(f"'{key}' must be between {lo} and {hi}.")
    return value


def _opt_bool(data: dict, key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValidationError(f"'{key}' must be true or false.")
    return value


def _str_list(data: dict, key: str, limit: int = 200) -> list[str] | None:
    value = data.get(key)
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) and 0 < len(v) <= 2000 for v in value):
        raise ValidationError(f"'{key}' must be a list of strings.")
    return value[:limit]


def _studio_scope(kb: KnowledgeBase, data: dict) -> set[str] | None:
    """Scope from ``sources``/``groups`` (lists) or the legacy single ``source`` focus."""
    sources = _str_list(data, "sources") or []
    single = _opt_str(data, "source", get_settings().source_name_max_chars)
    if single:
        if not kb.source_exists(single):
            raise NotFoundError("That source is no longer in your knowledge base. Pick another or clear the focus.")
        sources.append(single)
    return studio.scope_sources(kb, sources or None, _str_list(data, "groups"))


def _need_content(kb: KnowledgeBase):
    if kb.count() == 0:
        raise EmptyKnowledgeBase()


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ─── Application factory ────────────────────────────────────────────────────
def create_app(kb: KnowledgeBase | None = None) -> Flask:
    settings = get_settings()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app = Flask(
        __name__,
        template_folder=os.path.join(root, "templates"),
        static_folder=os.path.join(root, "static"),
    )
    app.secret_key = flask_secret(settings)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.config["JSON_SORT_KEYS"] = False
    app.json.sort_keys = False
    app.extensions["kf_rate_limiter"] = RateLimiter(settings.rate_limit_per_minute)
    if settings.trusted_proxies:
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=settings.trusted_proxies, x_proto=settings.trusted_proxies)

    if settings.cors_origins:
        from flask_cors import CORS

        CORS(app, resources={r"/api/*": {"origins": list(settings.cors_origins)}})

    if not settings.gemini_configured:
        logger.warning(MISSING_KEY_MESSAGE)

    def get_kb() -> KnowledgeBase:
        return kb if kb is not None else get_knowledge_base()

    app.extensions["kf_get_kb"] = get_kb

    @app.before_request
    def start_timer():
        g.request_id = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:12]
        g.started = time.perf_counter()

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; font-src 'self' data:; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        )
        resp.headers["X-Request-ID"] = g.get("request_id", "")
        if request.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-store")
            logger.info(
                "%s %s -> %s", request.method, request.path, resp.status_code,
                extra={
                    "request_id": g.get("request_id"), "path": request.path, "status": resp.status_code,
                    "duration_ms": round((time.perf_counter() - g.get("started", time.perf_counter())) * 1000, 1),
                },
            )
        return resp

    @app.errorhandler(413)
    def too_large(_exc):
        return _error(f"File too large. The limit is {settings.max_upload_mb} MB.", 413)

    @app.errorhandler(404)
    def not_found(_exc):
        return _error("Endpoint not found.", 404)

    @app.errorhandler(405)
    def bad_method(_exc):
        return _error("Method not allowed.", 405)

    # ── Pages & status ──────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template(
            "index.html", version=__version__, max_upload_mb=settings.max_upload_mb,
            max_question_chars=settings.max_question_chars,
        )

    @app.route("/api/health")
    def health():
        # Liveness only: no external calls, so a slow Gemini/Qdrant never fails it.
        return jsonify({"status": "ok", "service": "knowledgeforge", "version": __version__})

    @app.route("/api/ready")
    def ready():
        configured = get_settings().gemini_configured
        vector_ok = get_kb().health_check()
        redis_ok = redis_health_check()
        all_ready = configured and vector_ok and redis_ok
        return jsonify({
            "ready": all_ready,
            "gemini_configured": configured,
            "vector_store": get_settings().vector_store,
            "vector_store_healthy": vector_ok,
            "redis_healthy": redis_ok,
            "knowledge_base": get_kb().stats(),
        }), 200 if all_ready else 503

    # ── Knowledge base management ──────────────────────────────────────────
    @app.route("/api/get-sources")
    @api_endpoint
    def get_sources():
        k = get_kb()
        return jsonify({
            "success": True,
            "sources": k.list_sources(),
            "stats": k.stats(),
            "gemini_configured": get_settings().gemini_configured,
            "render_js": js_render.status(),
        })

    @app.route("/api/delete-source", methods=["POST"])
    @api_endpoint
    def delete_source():
        source = _required_str(_json_body(), "source", "Source", get_settings().source_name_max_chars)
        removed = get_kb().delete_source(source)
        if not removed:
            return _error("Source not found.", 404)
        return jsonify({"success": True, "message": f"Removed {removed} chunks.", "removed": removed})

    @app.route("/api/clear-data", methods=["POST"])
    @api_endpoint
    def clear_data():
        get_kb().clear()
        return jsonify({"success": True, "message": "Knowledge base cleared."})

    # ── Ingestion ───────────────────────────────────────────────────────────
    @app.route("/api/add-content", methods=["POST"])
    @api_endpoint(limited=True)
    def add_content():
        data = _json_body()
        source = _required_str(data, "source", "Source content", get_settings().max_text_chars)
        stype = (_opt_str(data, "source_type") or "").lower()
        if stype not in SOURCE_TYPES:
            raise ValidationError(f"Unknown source type '{stype}'. Use one of: {', '.join(SOURCE_TYPES)}.")
        render_js = _opt_str(data, "render_js", 10)
        ensure_gemini_configured()
        text, detail, effective_type = ingest_text_source(source, stype, render_js)
        if not text:
            raise IngestionError(detail or f"Failed to extract content from {effective_type}.")
        label = source
        if effective_type == "text":
            label = _note_label(source, _opt_str(data, "title", 200))
        return _ingest(get_kb(), text, label[: get_settings().source_name_max_chars], effective_type)

    @app.route("/api/upload-file", methods=["POST"])
    @api_endpoint(limited=True)
    def upload_file():
        f = request.files.get("file")
        if not f or not f.filename:
            raise ValidationError("No file provided.")
        filename = safe_filename(f.filename)
        data = f.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            return _error(f"File too large. The limit is {settings.max_upload_mb} MB.", 413)
        if not data:
            raise ValidationError("Uploaded file is empty.")
        detected = sniff_file_type(filename, f.mimetype or "", data[:64])
        if not detected:
            raise ValidationError(
                "Unsupported or mislabelled file. Use PDF, TXT/MD, audio (MP3/WAV/M4A/OGG), "
                "video (MP4/WEBM/MOV) or images (PNG/JPG/GIF/WEBP)."
            )
        source_type, mime = detected
        if source_type == "pdf":
            text = ingest_pdf_bytes(data)
        elif source_type == "text":
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
        else:
            ensure_gemini_configured()
            text = ingest_file_via_gemini(data, mime, FILE_PROMPTS[source_type])
        if not text:
            raise IngestionError(f"Could not extract any content from '{filename}'.")
        ensure_gemini_configured()
        return _ingest(get_kb(), text, filename, source_type, metadata={"filename": filename})

    # ── Q&A and chat ────────────────────────────────────────────────────────
    # ── Whole-site crawl ───────────────────────────────────────────────────
    @app.route("/api/crawl", methods=["GET"])
    @api_endpoint
    def crawl_list():
        return jsonify({"success": True, "jobs": get_crawl_manager().list()})

    @app.route("/api/crawl", methods=["POST"])
    @api_endpoint(limited=True)
    def crawl_start():
        data = _json_body()
        s = get_settings()
        url = _required_str(data, "url", "URL", 2000)
        render_js = resolve_render_mode(_opt_str(data, "render_js", 10))
        prefix = _opt_str(data, "path_prefix", 300) or "/"
        if not prefix.startswith("/"):
            raise ValidationError("'path_prefix' must start with '/'.")
        ensure_gemini_configured()
        job = get_crawl_manager().start(
            get_kb(), url,
            max_pages=_opt_int(data, "max_pages", min(s.crawl_site_default_pages, s.crawl_site_max_pages), 1,
                               s.crawl_site_max_pages),
            max_depth=_opt_int(data, "max_depth", min(2, s.crawl_site_max_depth), 0, s.crawl_site_max_depth),
            include_subdomains=_opt_bool(data, "include_subdomains", False),
            use_sitemap=_opt_bool(data, "use_sitemap", True),
            render_js=render_js,
            path_prefix=prefix,
        )
        return jsonify({"success": True, "job": job.snapshot()}), 202

    @app.route("/api/crawl/<job_id>", methods=["GET"])
    @api_endpoint
    def crawl_status(job_id):
        return jsonify({"success": True, "job": get_crawl_manager().get(job_id[:40]).snapshot()})

    @app.route("/api/crawl/<job_id>/cancel", methods=["POST"])
    @api_endpoint
    def crawl_cancel(job_id):
        job = get_crawl_manager().get(job_id[:40])
        job.cancel()
        return jsonify({"success": True, "job": job.snapshot(include_pages=False)})

    @app.route("/api/crawl/<job_id>/retry", methods=["POST"])
    @api_endpoint(limited=True)
    def crawl_retry(job_id):
        job = get_crawl_manager().get(job_id[:40])
        data = request.get_json(silent=True) or {}
        urls = data.get("urls") if isinstance(data, dict) else None
        if urls is not None and (not isinstance(urls, list) or not all(isinstance(u, str) for u in urls)):
            raise ValidationError("'urls' must be a list of strings.")
        n = job.retry(urls[:500] if urls else None)
        if not n:
            return _error("Nothing to retry.", 400)
        return jsonify({"success": True, "retrying": n, "job": job.snapshot(include_pages=False)})

    @app.route("/api/delete-group", methods=["POST"])
    @api_endpoint
    def delete_group():
        group = _required_str(_json_body(), "group", "Group", 300)
        removed, sources = get_kb().delete_group(group)
        if not removed:
            return _error("Group not found.", 404)
        return jsonify({"success": True, "removed": removed, "sources": sources,
                        "message": f"Removed {sources} pages ({removed} chunks)."})

    # ── Search tab ─────────────────────────────────────────────────────────
    @app.route("/api/search", methods=["POST"])
    @api_endpoint(limited=True)
    def search_api():
        data = _json_body()
        s = get_settings()
        query = _required_str(data, "query", "Search query", s.max_question_chars)
        filters = {}
        for key in ("sources", "sites", "types", "groups"):
            value = data.get(key)
            if value in (None, "", []):
                continue
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or not all(isinstance(v, str) and len(v) <= 2000 for v in value):
                raise ValidationError(f"'{key}' must be a list of strings.")
            filters[key] = set(value[:200])
        k = _opt_int(data, "k", 10, 1, 50)
        per_page = _opt_int(data, "per_page", 3, 1, 10)
        k_base = get_kb()
        if k_base.count() == 0:
            return _error("Knowledge base is empty. Add a web page, document or video first.", 422)
        ensure_gemini_configured()
        result = search_kb(k_base, query, k=k, per_page=per_page, **filters)
        return jsonify({"success": True, **result})

    @app.route("/api/ask-question", methods=["POST"])
    @api_endpoint(limited=True)
    def ask_question():
        data = _json_body()
        question = _required_str(data, "question", "Question", get_settings().max_question_chars)
        k = get_kb()
        source = _scope(k, data)
        ensure_gemini_configured()
        hits = k.hybrid_search(question, source_filter=source)
        if not hits:
            return jsonify({
                "success": True, "question": question, "answer": NO_CONTEXT_ANSWER, "sources": [], "grounded": False,
            })
        answer = generate_content(answer_prompt(question, hits), system=SYSTEM_INSTRUCTION)
        cites = _citations(hits, answer)
        return jsonify({
            "success": True, "question": question, "answer": answer, "sources": cites,
            "grounded": any(c["cited"] for c in cites),
        })

    def _chat_inputs():
        data = _json_body()
        message = _required_str(data, "message", "Message", get_settings().max_question_chars)
        k = get_kb()
        source = _scope(k, data)
        history = _clean_history(data.get("history"), get_settings().chat_history_turns)
        ensure_gemini_configured()
        hits = k.hybrid_search(message, source_filter=source) if k.count() else []
        return message, hits, chat_prompt(message, hits, history)

    @app.route("/api/chat", methods=["POST"])
    @api_endpoint(limited=True)
    def chat():
        message, hits, prompt = _chat_inputs()
        answer = generate_content(prompt, system=SYSTEM_INSTRUCTION)
        cites = _citations(hits, answer)
        return jsonify({
            "success": True, "message": message, "answer": answer, "sources": cites,
            "grounded": any(c["cited"] for c in cites),
        })

    @app.route("/api/chat/stream", methods=["POST"])
    @api_endpoint(limited=True)
    def chat_stream():
        """Server-sent events: ``meta`` (citations), many ``token``, then ``done`` or ``error``."""
        _message, hits, prompt = _chat_inputs()
        tokens = stream_content(prompt, system=SYSTEM_INSTRUCTION)

        def events():
            yield _sse("meta", {"sources": _citations(hits)})
            parts: list[str] = []
            try:
                for piece in tokens:
                    parts.append(piece)
                    yield _sse("token", {"text": piece})
            except KnowledgeForgeError as exc:
                yield _sse("error", {"message": str(exc), "status": exc.status})
                return
            except Exception:
                logger.exception("chat stream failed")
                yield _sse("error", {"message": "The answer stream was interrupted. Please try again.", "status": 500})
                return
            answer = "".join(parts)
            cites = _citations(hits, answer)
            yield _sse("done", {"answer": answer, "sources": cites, "grounded": any(c["cited"] for c in cites)})

        return Response(
            stream_with_context(events()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/chat/clear", methods=["POST"])
    def clear_chat():
        # History lives in the browser; kept for backwards compatibility.
        return jsonify({"success": True, "message": "Chat history cleared."})

    # ── Learning tools ─────────────────────────────────────────────────────
    @app.route("/api/summarize", methods=["POST"])
    @api_endpoint(limited=True)
    def summarize():
        k = get_kb()
        source = _scope(k, _json_body())
        pool = k.list_chunks(source)
        if not pool:
            return _error("No content to summarize yet. Add a source first.", 404)
        ensure_gemini_configured()
        text = _balanced_text(pool, get_settings().summary_max_chars)
        return jsonify({
            "success": True,
            "type": "summary",
            "title": f"Summary of {source}" if source else "Summary of all sources",
            "content": generate_content(summary_prompt(text), system=SYSTEM_INSTRUCTION),
        })

    @app.route("/api/generate-tool-content", methods=["POST"])
    @api_endpoint(limited=True)
    def generate_tool_content():
        data = _json_body()
        tool = _opt_str(data, "tool")
        if tool not in TOOLS:
            raise ValidationError(f"Invalid tool. Use one of: {', '.join(TOOLS)}.")
        tool = TOOL_ALIASES.get(tool, tool)
        topic = _required_str(data, "topic", "Topic", get_settings().tool_topic_max_chars)
        k = get_kb()
        source = _scope(k, data)
        if k.count() == 0:
            return _error("Knowledge base is empty. Add a web page, document or video first.", 422)
        ensure_gemini_configured()
        hits = k.hybrid_search(topic, source_filter=source)
        if not hits:
            return _error("No relevant content found for this topic. Try another topic or add sources.", 404)
        context = "\n\n".join(h["text"] for h in hits)
        prompt = tool_prompt(tool, topic, context)
        title = f"{tool.replace('_', ' ').title()}: {topic}"
        payload = {"success": True, "type": tool, "title": title, "sources": _citations(hits)}
        if tool in ("mindmap", "flowchart"):
            raw = generate_content(prompt, system=SYSTEM_INSTRUCTION)
            payload["mermaid_code"] = normalize_mermaid(raw, tool)
        elif tool in JSON_TOOLS:
            payload["content"] = _structured(tool, prompt)
        else:
            payload["content"] = generate_content(prompt, system=SYSTEM_INSTRUCTION)
        return jsonify(payload)

    @app.route("/api/extract-entities", methods=["POST"])
    @api_endpoint(limited=True)
    def extract_entities():
        k = get_kb()
        source = _scope(k, _json_body())
        pool = k.list_chunks(source)
        if not pool:
            return _error("No content available yet. Add a source first.", 404)
        ensure_gemini_configured()
        text = _balanced_text(pool, get_settings().entity_max_chars)
        raw = generate_content(entities_prompt(text), system=SYSTEM_INSTRUCTION, json_mode=True)
        try:
            entities = parse_json_output(raw)
        except LLMError:
            entities = {}
        if not isinstance(entities, dict):
            entities = {}
        clean = {
            str(cat)[:40]: sorted(
                {str(e).strip()[:120] for e in items if isinstance(e, (str, int, float)) and str(e).strip()}
            )
            for cat, items in entities.items()
            if isinstance(items, list)
        }
        return jsonify({"success": True, "entities": clean})

    # ── Create hub: storyboard + flashcards ───────────────────────────────
    @app.route("/api/storyboard", methods=["POST"])
    @api_endpoint(limited=True)
    def storyboard():
        data = _json_body()
        k = get_kb()
        topic = _opt_str(data, "topic", get_settings().tool_topic_max_chars)
        scope = _studio_scope(k, data)
        frames = _opt_int(data, "frames", 6, 2, 16)
        style = _opt_str(data, "style", 80) or "clear and friendly"
        _need_content(k)
        ensure_gemini_configured()
        result = create_mod.generate_storyboard(k, topic=topic, sources=scope, frames=frames, style=style)
        if not result or not result["frames"]:
            return _error("No relevant content found for this storyboard. Try another topic or add sources.", 404)
        return jsonify({"success": True, **result})

    @app.route("/api/storyboard/frame", methods=["POST"])
    @api_endpoint(limited=True)
    def storyboard_frame():
        data = _json_body()
        k = get_kb()
        frames = data.get("frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 30 or not all(isinstance(f, dict) for f in frames):
            raise ValidationError("'frames' must be the storyboard's list of frames.")
        frames = [{"title": studio.text_field(f.get("title"), 120)} for f in frames]
        index = _opt_int(data, "index", 0, 0, len(frames) - 1)
        topic = _opt_str(data, "topic", get_settings().tool_topic_max_chars)
        instruction = _opt_str(data, "instruction", 300)
        scope = _studio_scope(k, data)
        _need_content(k)
        ensure_gemini_configured()
        result = create_mod.regenerate_frame(k, frames=frames, index=index, topic=topic, instruction=instruction,
                                             sources=scope)
        if not result:
            return _error("No relevant content found for this frame.", 404)
        return jsonify({"success": True, "index": index, **result})

    @app.route("/api/flashcards", methods=["POST"])
    @api_endpoint(limited=True)
    def flashcards():
        data = _json_body()
        k = get_kb()
        topic = _opt_str(data, "topic", get_settings().tool_topic_max_chars)
        count = _opt_int(data, "count", 10, 1, 50)
        difficulty = _opt_str(data, "difficulty", 10) or "mixed"
        if difficulty not in create_mod.DIFFICULTIES:
            raise ValidationError(f"Difficulty must be one of: {', '.join(create_mod.DIFFICULTIES)}.")
        scope = _studio_scope(k, data)
        _need_content(k)
        ensure_gemini_configured()
        result = create_mod.generate_flashcards(k, topic=topic, sources=scope, count=count, difficulty=difficulty)
        if not result or not result["cards"]:
            return _error("Could not make flashcards from this content. Try another topic or add sources.", 404)
        return jsonify({"success": True, **result})

    # ── Knowledge graph and friends ───────────────────────────────────────
    def _graph_from(data):
        k = get_kb()
        scope = _studio_scope(k, data)
        topic = _opt_str(data, "topic", get_settings().tool_topic_max_chars)
        max_nodes = _opt_int(data, "max_nodes", 80, 5, graph_mod.MAX_NODES)
        _need_content(k)
        ensure_gemini_configured()
        g_ = graph_mod.build_graph(k, sources=scope, topic=topic, max_nodes=max_nodes,
                                   use_cache=not _opt_bool(data, "refresh", False))
        if not g_ or not g_["nodes"]:
            raise NotFoundError("No entities found in this content.")
        return k, scope, g_

    @app.route("/api/graph", methods=["POST"])
    @api_endpoint(limited=True)
    def graph():
        data = _json_body()
        _, _, g_ = _graph_from(data)
        payload = {"success": True, **g_}
        if _opt_bool(data, "include_map", False):
            payload["mermaid_code"] = graph_mod.concept_map(g_)
        return jsonify(payload)

    @app.route("/api/glossary", methods=["POST"])
    @api_endpoint(limited=True)
    def glossary():
        _, _, g_ = _graph_from(_json_body())
        return jsonify({"success": True, "terms": graph_mod.glossary(g_), "fallback": g_["fallback"]})

    @app.route("/api/concept-map", methods=["POST"])
    @api_endpoint(limited=True)
    def concept_map():
        _, _, g_ = _graph_from(_json_body())
        return jsonify({"success": True, "mermaid_code": graph_mod.concept_map(g_), "fallback": g_["fallback"]})

    @app.route("/api/timeline", methods=["POST"])
    @api_endpoint(limited=True)
    def timeline():
        k = get_kb()
        scope = _studio_scope(k, _json_body())
        _need_content(k)
        return jsonify({"success": True, "events": graph_mod.timeline(k, scope)})

    @app.route("/api/connect", methods=["POST"])
    @api_endpoint(limited=True)
    def connect():
        data = _json_body()
        a = _required_str(data, "a", "First concept", 120)
        b = _required_str(data, "b", "Second concept", 120)
        _, _, g_ = _graph_from(data)
        na, nb, paths = graph_mod.find_paths(g_, a, b)
        if not na or not nb:
            missing = a if not na else b
            return _error(f"'{missing}' is not in the knowledge graph. Pick a node from the graph.", 404)
        names = {n["id"]: n["name"] for n in g_["nodes"]}
        out = [{"nodes": p["nodes"], "names": [names[i] for i in p["nodes"]],
                "edges": [{k_: e[k_] for k_ in ("source", "target", "label", "sources")} for e in p["edges"]],
                "text": graph_mod.describe_path(g_, p)} for p in paths]
        fallback = True
        if out:
            facts = "\n".join(f"Path {i + 1}: {p['text']}" for i, p in enumerate(out))
            evidence = "\n".join(f"- {s['source']}: {s['snippet']}" for p in out[:1] for e in p["edges"]
                                  for s in e["sources"][:1])
            explanation = studio.llm_text(
                f"Explain in 2-4 sentences how '{na['name']}' connects to '{nb['name']}', using only these "
                f"graph paths and evidence. Do not invent facts.\n\n{facts}\n\nEvidence:\n{evidence}")
            fallback = explanation is None
            if not explanation:
                explanation = f"{na['name']} connects to {nb['name']} through: {out[0]['text']}"
                fallback = True
        else:
            explanation = f"No connection between {na['name']} and {nb['name']} was found in the graph."
        return jsonify({"success": True, "from": na["name"], "to": nb["name"], "paths": out,
                        "explanation": explanation, "fallback": fallback})

    @app.route("/api/related-sources", methods=["POST"])
    @api_endpoint(limited=True)
    def related_sources():
        data = _json_body()
        k = get_kb()
        source = _required_str(data, "source", "Source", get_settings().source_name_max_chars)
        if not k.source_exists(source):
            raise NotFoundError("That source is no longer in your knowledge base.")
        return jsonify({"success": True, "source": source,
                        "related": graph_mod.related_sources(k, source, _opt_int(data, "k", 5, 1, 20))})

    @app.route("/api/overview", methods=["POST"])
    @api_endpoint(limited=True)
    def overview():
        """Whole-KB summary plus a short summary and key topics per source."""
        data = _json_body()
        k = get_kb()
        scope = _studio_scope(k, data)
        _need_content(k)
        ensure_gemini_configured()
        chunks = [c for c in k.list_chunks() if scope is None or c["source"] in scope]
        all_texts = [c["text"] for c in k.list_chunks()]
        by_src: dict[str, list[dict]] = defaultdict(list)
        for c in chunks:
            by_src[c["source"]].append(c)
        per_source = [{"source": s, "chunks": len(cs), "key_topics": graph_mod.keywords([c["text"] for c in cs],
                                                                                         all_texts, 8)}
                      for s, cs in list(by_src.items())[:40]]
        summarize_each = _opt_bool(data, "per_source_summaries", True)
        fallback = False
        prompt = ("Summarise each numbered source in one or two sentences, then the whole collection in one "
                  'paragraph. Return ONLY JSON: {"sources": [{"n": int, "summary": str}], "overall": str, '
                  '"key_topics": [up to 8 short topics]}.\n\n')
        budget = max(1500, get_settings().summary_max_chars // max(1, len(per_source)))
        prompt += "\n\n".join(f'<source n="{i + 1}">\n{_balanced_text(by_src[p["source"]], budget)}\n</source>'
                               for i, p in enumerate(per_source))

        def validate(d):
            if not isinstance(d, dict) or not isinstance(d.get("overall"), str):
                return None
            return d

        got = studio.llm_json(prompt, validate, '{"sources": [{"n": int, "summary": str}], "overall": str, '
                                                '"key_topics": [str]}') if summarize_each else None
        if got:
            sums = {s.get("n"): studio.text_field(s.get("summary"), 600) for s in got.get("sources") or []
                    if isinstance(s, dict)}
            for i, p in enumerate(per_source):
                p["summary"] = sums.get(i + 1, "")
            overall = studio.text_field(got["overall"], 3000)
            topics = [studio.text_field(t, 60) for t in got.get("key_topics") or [] if isinstance(t, str)][:8]
        else:
            fallback = True
            for p in per_source:
                first = studio.sentences(by_src[p["source"]][0]["text"])
                p["summary"] = " ".join(first[:2])[:400]
            overall = (f"{len(per_source)} source(s) covering: " + ", ".join(graph_mod.keywords(
                [c["text"] for c in chunks], all_texts, 10)) + ".")
            topics = graph_mod.keywords([c["text"] for c in chunks], all_texts, 8)
        return jsonify({"success": True, "overall": overall, "key_topics": topics, "sources": per_source,
                        "fallback": fallback})

    @app.route("/api/cluster-topics", methods=["GET"])
    @api_endpoint(limited=True)
    def cluster_topics():
        k = get_kb()
        minimum = get_settings().cluster_min_chunks
        if k.count() < minimum:
            return _error(f"Need at least {minimum} content chunks to cluster. Add more content first.", 400)
        ensure_gemini_configured()
        chunks, embeddings = k.get_embeddings_for_clustering()
        image, labels = render_cluster_plot(chunks, embeddings, get_settings().cluster_max_topics)
        return jsonify({"success": True, "image": image, "labels": labels})

    return app


def render_cluster_plot(chunks: list[dict], embeddings: np.ndarray, max_topics: int = 5) -> tuple[str, list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n_clusters = max(2, min(max_topics, embeddings.shape[0] // 2))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings)
    labels = []
    for i in range(n_clusters):
        members = np.where(km.labels_ == i)[0][:3]
        sample = " ".join(chunks[int(j)]["text"] for j in members)
        try:
            labels.append(clean_label(generate_content(cluster_label_prompt(sample))))
        except KnowledgeForgeError:
            labels.append(f"Topic {i + 1}")
    reduced = PCA(n_components=2).fit_transform(embeddings)

    palette = [
        "#4f46e5", "#0891b2", "#059669", "#d97706", "#dc2626", "#7c3aed", "#db2777", "#65a30d", "#0d9488", "#ea580c",
    ]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i in range(n_clusters):
        mask = km.labels_ == i
        ax.scatter(reduced[mask, 0], reduced[mask, 1], s=90, alpha=0.85, color=palette[i % len(palette)],
                   edgecolors="white", linewidths=0.6, label=labels[i])
    ax.set_title("Topic clusters in your knowledge base", fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.2)
    ax.legend(title="Topics", loc="best", frameon=True)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}", labels


configure_logging()
