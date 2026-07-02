"""KnowledgeForge V2 — production Flask application."""
import base64
import io
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from flask import Flask, jsonify, render_template, request, session
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import (
    CHAT_HISTORY_TURNS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FLASK_SECRET,
    GEMINI_API_KEY,
    MAX_UPLOAD_BYTES,
    REDIS_URL,
    SESSION_TYPE,
    VECTOR_STORE,
)
from services.cache import redis_health_check
from services.embeddings import ensure_gemini_configured
from services.ingestion import (
    chunk_text,
    detect_file_type,
    ingest_file_via_gemini,
    ingest_pdf_bytes,
    ingest_text_source,
)
from services.knowledge_base import KnowledgeBase
from services.llm_utils import (
    clean_llm_output,
    generate_content,
    parse_json_output,
    sanitize_mermaid,
)

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("knowledgeforge")

# ─── App Init ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = FLASK_SECRET
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["SESSION_PERMANENT"] = False

if SESSION_TYPE == "redis" and REDIS_URL:
    import redis
    from flask_session import Session
    app.config["SESSION_TYPE"] = "redis"
    app.config["SESSION_REDIS"] = redis.from_url(REDIS_URL)
    app.config["SESSION_USE_SIGNER"] = True
    app.config["SESSION_KEY_PREFIX"] = "kf:session:"
    Session(app)
    logger.info("Flask sessions backed by Redis.")
else:
    app.config["SESSION_TYPE"] = "filesystem"
    logger.info("Flask sessions using filesystem (single-instance).")

CORS(app)

kb = KnowledgeBase()


# ─── Error Handlers ──────────────────────────────────────────────────────────
@app.errorhandler(413)
def request_too_large(_exc):
    return jsonify({"success": False, "message": "File too large. Reduce upload size and try again."}), 413


@app.errorhandler(404)
def not_found(_exc):
    return jsonify({"success": False, "message": "Endpoint not found."}), 404


@app.errorhandler(500)
def internal_error(exc):
    logger.exception("Unhandled error: %s", exc)
    return jsonify({"success": False, "message": "An internal error occurred. Please try again."}), 500


def _require_json():
    data = request.get_json(silent=True)
    if data is None:
        return None, (jsonify({"success": False, "message": "Invalid or missing JSON body."}), 400)
    return data, None


# ─── Health & Status ─────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "knowledgeforge-v2"})


@app.route("/api/ready")
def ready():
    configured = bool(GEMINI_API_KEY)
    vector_ok = kb.health_check()
    redis_ok = redis_health_check()
    embed_ok = False
    embed_model = None
    if configured:
        try:
            from services.embeddings import _resolve_active_model
            embed_model = _resolve_active_model()
            embed_ok = True
        except Exception as exc:
            logger.warning("Embedding model probe failed: %s", exc)
    stats = kb.stats()
    all_ready = configured and vector_ok and redis_ok and embed_ok
    return jsonify({
        "ready": all_ready,
        "gemini_configured": configured,
        "embedding_model": embed_model,
        "embedding_healthy": embed_ok,
        "vector_store": VECTOR_STORE,
        "vector_store_healthy": vector_ok,
        "redis_healthy": redis_ok,
        "knowledge_base": stats,
    }), 200 if all_ready else 503


# ─── Pages ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─── Content Ingestion ───────────────────────────────────────────────────────
@app.route("/api/add-content", methods=["POST"])
def add_content():
    data, err = _require_json()
    if err:
        return err

    source = (data.get("source") or "").strip()
    stype = (data.get("source_type") or "").strip()
    if not source:
        return jsonify({"success": False, "message": "Source content is required."}), 400

    try:
        ensure_gemini_configured()
        text, detail, effective_type = ingest_text_source(source, stype)
        if detail.startswith("Unknown source type"):
            return jsonify({"success": False, "message": detail}), 400

        if not text:
            msg = detail or f"Failed to extract content from {effective_type}."
            return jsonify({"success": False, "message": msg}), 422

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return jsonify({"success": False, "message": "No usable text found in content."}), 422

        success = kb.add_chunks(chunks, source, effective_type)
        note = ""
        if stype == "web" and effective_type == "youtube":
            note = " (detected as YouTube — use the YouTube tab next time)"
        return jsonify({
            "success": success,
            "message": (
                f"Added {len(chunks)} chunks from {effective_type}.{note}"
                if success else "Failed to process content."
            ),
            "chunks_added": len(chunks) if success else 0,
            "source_type": effective_type,
        })
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:
        logger.exception("add-content failed: %s", exc)
        return jsonify({"success": False, "message": "Failed to ingest content."}), 500


@app.route("/api/upload-file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file provided."}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"success": False, "message": "Empty file upload."}), 400

    filename = f.filename
    data = f.read()
    if not data:
        return jsonify({"success": False, "message": "Uploaded file is empty."}), 400

    mime_type = f.mimetype or "application/octet-stream"
    detected = detect_file_type(filename, mime_type)
    if not detected:
        return jsonify({"success": False, "message": f"Unsupported file type: {mime_type}"}), 415

    source_type, effective_mime = detected

    try:
        ensure_gemini_configured()
        if source_type == "pdf":
            text = ingest_pdf_bytes(data)
        elif source_type == "text":
            text = data.decode("utf-8", errors="replace")
        elif source_type == "audio":
            text = ingest_file_via_gemini(
                data, effective_mime,
                "Transcribe this audio lecture completely and accurately. Preserve all key concepts and details.",
            )
        elif source_type == "video":
            text = ingest_file_via_gemini(
                data, effective_mime,
                "Analyze this educational video. Transcribe all speech and describe key visual content, diagrams, and concepts shown.",
            )
        elif source_type == "image":
            text = ingest_file_via_gemini(
                data, effective_mime,
                "Extract all text (OCR) from this image. Also describe any diagrams, charts, or visual content in detail.",
            )
        else:
            return jsonify({"success": False, "message": "Unsupported file type."}), 415

        if not text:
            return jsonify({"success": False, "message": "Failed to extract content from file."}), 422

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return jsonify({"success": False, "message": "No usable text extracted from file."}), 422

        success = kb.add_chunks(chunks, filename, source_type, metadata={"filename": filename})
        return jsonify({
            "success": success,
            "message": f"Added {len(chunks)} chunks from '{filename}'." if success else "Failed to process file.",
            "chunks_added": len(chunks) if success else 0,
        })
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:
        logger.exception("upload-file failed: %s", exc)
        return jsonify({"success": False, "message": "Failed to process uploaded file."}), 500


# ─── Q&A & Chat ──────────────────────────────────────────────────────────────
@app.route("/api/ask-question", methods=["POST"])
def ask_question():
    data, err = _require_json()
    if err:
        return err

    question = (data.get("question") or "").strip()
    source = data.get("source") or None
    if not question:
        return jsonify({"success": False, "message": "Question is required."}), 400

    try:
        ensure_gemini_configured()
        chunks = kb.hybrid_search(question, source_filter=source)
        if not chunks:
            return jsonify({
                "success": True,
                "question": question,
                "answer": "I don't have enough information to answer that. Please add relevant content first.",
            })

        context = "\n\n---\n\n".join(chunks)
        prompt = (
            "You are a helpful educational AI assistant. Use ONLY the provided context to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, well-structured answer in markdown. "
            "If the context doesn't contain the answer, say so honestly."
        )
        answer = generate_content(prompt)
        return jsonify({"success": True, "question": question, "answer": answer})
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:
        logger.exception("ask-question failed: %s", exc)
        return jsonify({"success": False, "message": "Failed to generate answer."}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data, err = _require_json()
    if err:
        return err

    message = (data.get("message") or "").strip()
    source = data.get("source") or None
    if not message:
        return jsonify({"success": False, "message": "Message is required."}), 400

    if "chat_history" not in session:
        session["chat_history"] = []

    try:
        ensure_gemini_configured()
        chunks = kb.hybrid_search(message, source_filter=source)
        context = "\n\n---\n\n".join(chunks) if chunks else "No specific context retrieved."

        history_text = "\n".join(
            f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
            for h in session["chat_history"][-CHAT_HISTORY_TURNS:]
        )

        prompt = (
            "You are a knowledgeable educational AI assistant. Use the context below to help the user.\n\n"
            f"Knowledge Context:\n{context}\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"User: {message}\n\n"
            "Respond clearly and helpfully. Use markdown where appropriate."
        )
        answer = generate_content(prompt)

        session["chat_history"].append({"role": "user", "content": message})
        session["chat_history"].append({"role": "assistant", "content": answer})
        session.modified = True

        return jsonify({
            "success": True,
            "message": message,
            "answer": answer,
            "history": session["chat_history"],
        })
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:
        logger.exception("chat failed: %s", exc)
        return jsonify({"success": False, "message": "Failed to process chat message."}), 500


@app.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    session.pop("chat_history", None)
    return jsonify({"success": True, "message": "Chat history cleared."})


# ─── Learning Tools ──────────────────────────────────────────────────────────
@app.route("/api/summarize", methods=["POST"])
def summarize():
    data, err = _require_json()
    if err:
        return err

    source = data.get("source") or None
    pool = kb.list_chunks(source)
    if not pool:
        return jsonify({"success": False, "message": "No content to summarize."}), 404

    try:
        ensure_gemini_configured()
        texts = " ".join(c["text"] for c in pool[:30])
        prompt = (
            "Create a comprehensive, well-structured summary of the following educational content. "
            "Use headings, bullet points, and key takeaways. Format in markdown.\n\n"
            f"{texts}"
        )
        content = generate_content(prompt)
        return jsonify({
            "success": True,
            "type": "summary",
            "title": f"Summary{' of ' + source if source else ''}",
            "content": content,
        })
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:
        logger.exception("summarize failed: %s", exc)
        return jsonify({"success": False, "message": "Failed to generate summary."}), 500


@app.route("/api/generate-tool-content", methods=["POST"])
def generate_tool_content():
    data, err = _require_json()
    if err:
        return err

    tool = data.get("tool")
    topic = (data.get("topic") or "").strip()
    source = data.get("source") or None

    if not topic:
        return jsonify({"success": False, "message": "Enter a topic in the input field first."}), 400

    if tool not in ("mock_test", "mindmap", "flowchart", "storyboard", "flashcards", "study_plan"):
        return jsonify({"success": False, "message": "Invalid tool specified."}), 400

    if kb.count() == 0:
        return jsonify({
            "success": False,
            "message": "Knowledge base is empty. Add a web page, YouTube video, or text first.",
        }), 422

    try:
        ensure_gemini_configured()
        chunks = kb.hybrid_search(topic, source_filter=source)
        if not chunks:
            return jsonify({
                "success": False,
                "message": "No relevant content found for this topic. Try a different topic or add more sources.",
            }), 404

        context = "\n\n".join(chunks)
        mermaid_rules = (
            "MERMAID RULES: 1) Use simple single-word IDs (A, B, C1). "
            '2) Wrap node text with special chars in double quotes: A["Text (with parens)"].'
        )

        prompts = {
            "mock_test": (
                f"Based on the context, generate a 5-question multiple-choice test on '{topic}'. "
                f"Provide 4 options (A-D) per question. End with an 'Answers:' section.\n\nContext:\n{context}"
            ),
            "mindmap": (
                f"Create a Mermaid.js 'mindmap' syntax about '{topic}' from the context.\n"
                f"{mermaid_rules}\nContext:\n{context}\nOutput ONLY ```mermaid ... ```"
            ),
            "flowchart": (
                f"Create a Mermaid.js 'graph TD' flowchart for '{topic}' from the context.\n"
                f"{mermaid_rules}\nContext:\n{context}\nOutput ONLY ```mermaid ... ```"
            ),
            "storyboard": (
                f"Create a 3-panel storyboard about '{topic}'. For each panel give 'scene' (description) "
                f"and 'icon' (Font Awesome class like 'fas fa-lightbulb'). "
                f"Return ONLY a valid JSON array inside ```json ... ```.\n\nContext:\n{context}"
            ),
            "flashcards": (
                f"Generate 8 flashcards about '{topic}' from the context. "
                f'Each flashcard has a "question" (the concept/term) and "answer" (explanation). '
                f'Return ONLY a JSON array: [{{"question":"...","answer":"..."}}] inside ```json ... ```.\n\nContext:\n{context}'
            ),
            "study_plan": (
                f"Create a structured 4-week study plan to master '{topic}' based on the context. "
                f"Format as markdown with week headings, daily goals, and resources. "
                f"Be specific and actionable.\n\nContext:\n{context}"
            ),
        }

        resp_text = generate_content(prompts[tool])

        if tool in ("mindmap", "flowchart"):
            raw = clean_llm_output(resp_text, "mermaid")
            safe = sanitize_mermaid(raw)
            return jsonify({
                "success": True,
                "type": tool,
                "title": f"{tool.replace('_', ' ').title()} — {topic}",
                "mermaid_code": safe,
            })

        if tool in ("storyboard", "flashcards"):
            try:
                content = parse_json_output(resp_text)
                return jsonify({
                    "success": True,
                    "type": tool,
                    "title": f"{tool.replace('_', ' ').title()} — {topic}",
                    "content": content,
                })
            except Exception:
                return jsonify({"success": False, "message": "AI returned invalid JSON. Please try again."}), 502

        return jsonify({
            "success": True,
            "type": tool,
            "title": f"{tool.replace('_', ' ').title()} — {topic}",
            "content": resp_text,
        })
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:
        logger.exception("generate-tool-content failed: %s", exc)
        return jsonify({"success": False, "message": "Failed to generate tool content."}), 500


@app.route("/api/extract-entities", methods=["POST"])
def extract_entities():
    data, err = _require_json()
    if err:
        return err

    source = data.get("source") or None
    text = " ".join(c["text"] for c in kb.list_chunks(source))
    if not text:
        return jsonify({"success": False, "message": "No content available."}), 404

    try:
        ensure_gemini_configured()
        prompt = (
            "Extract named entities from the text below. "
            "Return a JSON object with keys 'Person', 'Organization', 'Location', 'Concept' — "
            "each containing a list of unique names. Use ```json ... ```.\n\n"
            f"Text:\n{text[:4000]}"
        )
        entities = parse_json_output(generate_content(prompt))
        return jsonify({"success": True, "entities": entities})
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception:
        return jsonify({"success": False, "message": "Failed to parse entities."}), 502


@app.route("/api/cluster-topics", methods=["GET"])
def cluster_topics():
    if kb.count() < 5:
        return jsonify({"success": False, "message": "Need at least 5 content chunks to cluster."}), 400

    try:
        ensure_gemini_configured()
        chunk_list, embeddings = kb.get_embeddings_for_clustering()
        if embeddings.shape[0] < 5:
            return jsonify({"success": False, "message": "Need at least 5 content chunks to cluster."}), 400

        num_clusters = min(5, embeddings.shape[0] // 2)
        if num_clusters < 2:
            return jsonify({"success": False, "message": "Not enough diverse content."}), 400

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(embeddings)
        labels = kmeans.labels_
        cluster_labels = []
        for i in range(num_clusters):
            sample = " ".join(chunk_list[int(j)]["text"] for j in np.where(labels == i)[0][:3])
            label = generate_content(f"Give a concise 2-3 word topic label for: {sample[:500]}")
            cluster_labels.append(label.replace('"', "").replace("*", ""))

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
        palette = ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"]
        sns.scatterplot(
            x=reduced[:, 0], y=reduced[:, 1],
            hue=[cluster_labels[idx] for idx in labels],
            palette=palette[:num_clusters], s=120, alpha=0.85, ax=ax,
            edgecolor="white", linewidth=0.5,
        )
        ax.set_title("Content Topic Clusters", fontsize=16, fontweight="bold", color="#1e293b")
        ax.set_xlabel("Component 1", color="#475569")
        ax.set_ylabel("Component 2", color="#475569")
        ax.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left")
        fig.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
        })
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 503
    except Exception as exc:
        logger.exception("cluster-topics failed: %s", exc)
        return jsonify({"success": False, "message": "Failed to cluster topics."}), 500


# ─── Knowledge Base Management ───────────────────────────────────────────────
@app.route("/api/get-sources", methods=["GET"])
def get_sources():
    sources = []
    seen = set()
    for c in kb.list_chunks():
        if c["source"] not in seen:
            seen.add(c["source"])
            sources.append({"source": c["source"], "type": c.get("source_type", "text")})
    return jsonify({"success": True, "sources": sources, "stats": kb.stats()})


@app.route("/api/clear-data", methods=["POST"])
def clear_data():
    kb.clear()
    session.pop("chat_history", None)
    return jsonify({"success": True, "message": "Knowledge base cleared."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)