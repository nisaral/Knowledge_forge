"""Flask application factory and JSON API for KnowledgeForge."""
from __future__ import annotations

import base64
import functools
import io
import logging
import os

import numpy as np
from flask import Flask, jsonify, render_template, request

from knowledge_forge import __version__
from knowledge_forge.cache import redis_health_check
from knowledge_forge.chunking import chunk_text, normalize_text
from knowledge_forge.config import flask_secret, get_settings
from knowledge_forge.errors import (
    MISSING_KEY_MESSAGE,
    IngestionError,
    KnowledgeForgeError,
    ValidationError,
)
from knowledge_forge.ingestion import (
    SOURCE_TYPES,
    detect_file_type,
    ingest_file_via_gemini,
    ingest_pdf_bytes,
    ingest_text_source,
)
from knowledge_forge.knowledge_base import KnowledgeBase, get_knowledge_base
from knowledge_forge.llm import (
    clean_label,
    clean_llm_output,
    ensure_gemini_configured,
    generate_content,
    parse_json_output,
    sanitize_mermaid,
)
from knowledge_forge.prompts import (
    TOOLS,
    answer_prompt,
    chat_prompt,
    cluster_label_prompt,
    entities_prompt,
    summary_prompt,
    tool_prompt,
)

logger = logging.getLogger("knowledgeforge")

FILE_PROMPTS = {
    "audio": "Transcribe this audio completely and accurately. Preserve all key concepts and details.",
    "video": (
        "Analyze this educational video. Transcribe all speech and describe key visual content, "
        "diagrams, and concepts shown."
    ),
    "image": "Extract all text (OCR) from this image. Also describe any diagrams, charts, or visual content in detail.",
}

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in your knowledge base. Add a document, web page or "
    "video on this topic first, or try rephrasing the question."
)


# ─── Helpers ────────────────────────────────────────────────────────────────
def _error(message: str, status: int):
    return jsonify({"success": False, "message": message}), status


def _json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValidationError("Invalid or missing JSON body.")
    return data


def _opt_str(data: dict, key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"'{key}' must be a string.")
    return value.strip() or None


def _required_str(data: dict, key: str, label: str, max_len: int | None = None) -> str:
    value = _opt_str(data, key)
    if not value:
        raise ValidationError(f"{label} is required.")
    if max_len and len(value) > max_len:
        raise ValidationError(f"{label} is too long (max {max_len} characters).")
    return value


def _citations(hits: list[dict]) -> list[dict]:
    return [
        {
            "n": i,
            "source": h.get("source", ""),
            "source_type": h.get("source_type", "text"),
            "snippet": (h.get("text") or "")[:320],
            "score": h.get("score"),
        }
        for i, h in enumerate(hits, start=1)
    ]


def _clean_history(raw, turns: int) -> list[dict]:
    """Validate client-supplied chat history and keep the last ``turns`` exchanges."""
    if not isinstance(raw, list) or turns <= 0:
        return []
    cleaned = []
    for item in raw[-turns * 2:]:
        if not isinstance(item, dict):
            continue
        role = "user" if item.get("role") == "user" else "assistant"
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()[:2000]})
    return cleaned


def api_endpoint(fn):
    """Translate domain errors into consistent JSON responses."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KnowledgeForgeError as exc:
            if exc.status >= 500:
                logger.warning("%s failed: %s", fn.__name__, exc)
            return _error(str(exc), exc.status)
        except Exception:
            logger.exception("%s failed unexpectedly", fn.__name__)
            return _error("Something went wrong on the server. Please try again.", 500)

    return wrapper


def _ingest(kb: KnowledgeBase, text: str | None, source: str, source_type: str, metadata=None):
    settings = get_settings()
    text = normalize_text(text or "")
    if not text:
        raise IngestionError("No readable text could be extracted from this source.")
    truncated = len(text) > settings.max_text_chars
    text = text[: settings.max_text_chars]
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    added = kb.add_chunks(chunks, source, source_type, metadata=metadata)
    if added:
        message = f"Added {added} chunk{'s' if added != 1 else ''} from {source_type}."
    else:
        message = "This content is already in your knowledge base; nothing new to add."
    if truncated:
        message += f" (Content was truncated to {settings.max_text_chars:,} characters.)"
    return jsonify({
        "success": True,
        "message": message,
        "chunks_added": added,
        "chunks_total": len(chunks),
        "source": source,
        "source_type": source_type,
    })


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

    if settings.cors_origins:
        from flask_cors import CORS

        CORS(app, resources={r"/api/*": {"origins": list(settings.cors_origins)}})

    if not settings.gemini_configured:
        logger.warning(MISSING_KEY_MESSAGE)

    def get_kb() -> KnowledgeBase:
        return kb if kb is not None else get_knowledge_base()

    app.extensions["kf_get_kb"] = get_kb

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
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
        return render_template("index.html", version=__version__, max_upload_mb=settings.max_upload_mb)

    @app.route("/api/health")
    def health():
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
        })

    @app.route("/api/delete-source", methods=["POST"])
    @api_endpoint
    def delete_source():
        source = _required_str(_json_body(), "source", "Source")
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
    @api_endpoint
    def add_content():
        data = _json_body()
        source = _required_str(data, "source", "Source content")
        stype = (_opt_str(data, "source_type") or "").lower()
        if stype not in SOURCE_TYPES:
            raise ValidationError(f"Unknown source type '{stype}'. Use one of: {', '.join(SOURCE_TYPES)}.")
        ensure_gemini_configured()
        text, detail, effective_type = ingest_text_source(source, stype)
        if not text:
            raise IngestionError(detail or f"Failed to extract content from {effective_type}.")
        label = source
        if effective_type == "text":
            title = _opt_str(data, "title")
            label = title or f"Note: {normalize_text(source)[:48]}"
        return _ingest(get_kb(), text, label[:500], effective_type)

    @app.route("/api/upload-file", methods=["POST"])
    @api_endpoint
    def upload_file():
        f = request.files.get("file")
        if not f or not f.filename:
            raise ValidationError("No file provided.")
        filename = os.path.basename(f.filename.replace("\\", "/"))[:200] or "upload"
        data = f.read()
        if not data:
            raise ValidationError("Uploaded file is empty.")
        detected = detect_file_type(filename, f.mimetype or "")
        if not detected:
            raise ValidationError(
                "Unsupported file type. Use PDF, TXT/MD, audio (MP3/WAV/M4A/OGG), "
                "video (MP4/WEBM/MOV) or images (PNG/JPG/GIF/WEBP)."
            )
        source_type, mime = detected
        ensure_gemini_configured()
        if source_type == "pdf":
            text = ingest_pdf_bytes(data)
        elif source_type == "text":
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
        else:
            text = ingest_file_via_gemini(data, mime, FILE_PROMPTS[source_type])
        if not text:
            raise IngestionError(f"Could not extract any content from '{filename}'.")
        return _ingest(get_kb(), text, filename, source_type, metadata={"filename": filename})

    # ── Q&A and chat ────────────────────────────────────────────────────────
    @app.route("/api/ask-question", methods=["POST"])
    @api_endpoint
    def ask_question():
        data = _json_body()
        question = _required_str(data, "question", "Question", get_settings().max_question_chars)
        source = _opt_str(data, "source")
        ensure_gemini_configured()
        hits = get_kb().hybrid_search(question, source_filter=source)
        if not hits:
            return jsonify({"success": True, "question": question, "answer": NO_CONTEXT_ANSWER, "sources": []})
        answer = generate_content(answer_prompt(question, hits))
        return jsonify({"success": True, "question": question, "answer": answer, "sources": _citations(hits)})

    @app.route("/api/chat", methods=["POST"])
    @api_endpoint
    def chat():
        data = _json_body()
        message = _required_str(data, "message", "Message", get_settings().max_question_chars)
        source = _opt_str(data, "source")
        history = _clean_history(data.get("history"), get_settings().chat_history_turns)
        ensure_gemini_configured()
        hits = get_kb().hybrid_search(message, source_filter=source) if get_kb().count() else []
        answer = generate_content(chat_prompt(message, hits, history))
        return jsonify({"success": True, "message": message, "answer": answer, "sources": _citations(hits)})

    @app.route("/api/chat/clear", methods=["POST"])
    def clear_chat():
        # History lives in the browser; kept for backwards compatibility.
        return jsonify({"success": True, "message": "Chat history cleared."})

    # ── Learning tools ─────────────────────────────────────────────────────
    @app.route("/api/summarize", methods=["POST"])
    @api_endpoint
    def summarize():
        source = _opt_str(_json_body(), "source")
        pool = get_kb().list_chunks(source)
        if not pool:
            return _error("No content to summarize yet. Add a source first.", 404)
        ensure_gemini_configured()
        text = " ".join(c["text"] for c in pool[:30])
        return jsonify({
            "success": True,
            "type": "summary",
            "title": f"Summary of {source}" if source else "Summary of all sources",
            "content": generate_content(summary_prompt(text)),
        })

    @app.route("/api/generate-tool-content", methods=["POST"])
    @api_endpoint
    def generate_tool_content():
        data = _json_body()
        tool = _opt_str(data, "tool")
        topic = _required_str(data, "topic", "Topic", 300)
        source = _opt_str(data, "source")
        if tool not in TOOLS:
            raise ValidationError("Invalid tool specified.")
        k = get_kb()
        if k.count() == 0:
            return _error("Knowledge base is empty. Add a web page, document or video first.", 422)
        ensure_gemini_configured()
        hits = k.hybrid_search(topic, source_filter=source)
        if not hits:
            return _error("No relevant content found for this topic. Try another topic or add sources.", 404)
        context = "\n\n".join(h["text"] for h in hits)
        raw = generate_content(tool_prompt(tool, topic, context))
        title = f"{tool.replace('_', ' ').title()}: {topic}"
        payload = {"success": True, "type": tool, "title": title, "sources": _citations(hits)}
        if tool in ("mindmap", "flowchart"):
            payload["mermaid_code"] = sanitize_mermaid(clean_llm_output(raw, "mermaid"))
        elif tool in ("storyboard", "flashcards"):
            content = parse_json_output(raw)
            if not isinstance(content, list):
                raise IngestionError("The model returned an unexpected format. Please try again.")
            payload["content"] = [c for c in content if isinstance(c, dict)]
        else:
            payload["content"] = raw
        return jsonify(payload)

    @app.route("/api/extract-entities", methods=["POST"])
    @api_endpoint
    def extract_entities():
        source = _opt_str(_json_body(), "source")
        text = " ".join(c["text"] for c in get_kb().list_chunks(source))
        if not text:
            return _error("No content available yet. Add a source first.", 404)
        ensure_gemini_configured()
        entities = parse_json_output(generate_content(entities_prompt(text[:6000])))
        if not isinstance(entities, dict):
            entities = {}
        clean = {
            str(cat): sorted({str(e).strip() for e in items if str(e).strip()})
            for cat, items in entities.items()
            if isinstance(items, list)
        }
        return jsonify({"success": True, "entities": clean})

    @app.route("/api/cluster-topics", methods=["GET"])
    @api_endpoint
    def cluster_topics():
        k = get_kb()
        if k.count() < 5:
            return _error("Need at least 5 content chunks to cluster. Add more content first.", 400)
        ensure_gemini_configured()
        chunks, embeddings = k.get_embeddings_for_clustering()
        image, labels = render_cluster_plot(chunks, embeddings)
        return jsonify({"success": True, "image": image, "labels": labels})

    return app


def render_cluster_plot(chunks: list[dict], embeddings: np.ndarray) -> tuple[str, list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n_clusters = max(2, min(5, embeddings.shape[0] // 2))
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

    palette = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#dc2626"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i in range(n_clusters):
        mask = km.labels_ == i
        ax.scatter(reduced[mask, 0], reduced[mask, 1], s=90, alpha=0.85, color=palette[i],
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


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


_configure_logging()

