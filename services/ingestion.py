"""Multimodal content ingestion helpers."""
import logging
import mimetypes
import os
import re
import tempfile
import time

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

from config import (
    CRAWL_TIMEOUT_SECONDS,
    GEMINI_FILE_MAX_POLLS,
    GEMINI_FILE_POLL_SECONDS,
    GEN_MODEL,
)
from services.embeddings import ensure_gemini_configured

logger = logging.getLogger(__name__)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window word chunker with overlap."""
    words = re.split(r"\s+", text.strip())
    if not words or (len(words) == 1 and not words[0]):
        return []
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def crawl_url(url: str) -> str | None:
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "KnowledgeForge/2.0"},
            timeout=CRAWL_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
            el.decompose()
        text = " ".join(soup.stripped_strings)
        return text if text.strip() else None
    except Exception as exc:
        logger.error("Error crawling %s: %s", url, exc)
        return None


def _extract_youtube_id(video_url: str) -> str | None:
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(video_url: str) -> str | None:
    vid_id = _extract_youtube_id(video_url)
    if not vid_id:
        return None
    try:
        # youtube-transcript-api < 1.0
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript = YouTubeTranscriptApi.get_transcript(vid_id)
            return " ".join(d["text"] for d in transcript)
        # youtube-transcript-api >= 1.0
        fetched = YouTubeTranscriptApi().fetch(vid_id)
        return " ".join(s.text for s in fetched.snippets)
    except Exception as exc:
        logger.error("Error fetching YouTube transcript for %s: %s", vid_id, exc)
        return None


def ingest_pdf_bytes(data: bytes) -> str | None:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        texts = [page.get_text("text") for page in doc]
        raw = "\n".join(texts).strip()
        if len(raw) < 100:
            return ingest_pdf_via_gemini_bytes(data)
        return raw
    except ImportError:
        return ingest_pdf_via_gemini_bytes(data)
    except Exception as exc:
        logger.error("PDF extraction error: %s", exc)
        return ingest_pdf_via_gemini_bytes(data)


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
        uploaded = genai.upload_file(tmp_path, mime_type=mime_type)
        for _ in range(GEMINI_FILE_MAX_POLLS):
            f = genai.get_file(uploaded.name)
            state = f.state.name if hasattr(f.state, "name") else str(f.state)
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini file processing failed.")
            time.sleep(GEMINI_FILE_POLL_SECONDS)
        else:
            raise TimeoutError("Gemini file processing timed out.")

        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content([uploaded, prompt])
        return (resp.text or "").strip() or None
    except Exception as exc:
        logger.error("Gemini file ingestion error (%s): %s", mime_type, exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if uploaded:
            try:
                genai.delete_file(uploaded.name)
            except Exception:
                pass


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
    if lower.endswith(".txt") or mime_type == "text/plain":
        return "text", "text/plain"
    return None