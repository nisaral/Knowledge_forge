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

from config import (
    CRAWL_TIMEOUT_SECONDS,
    GEMINI_FILE_MAX_POLLS,
    GEMINI_FILE_POLL_SECONDS,
    GEN_MODEL,
)
from services.embeddings import ensure_gemini_configured

logger = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


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


def is_youtube_url(url: str) -> bool:
    return _extract_youtube_id(url) is not None


def crawl_url(url: str) -> str | None:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    if is_youtube_url(url):
        logger.info("YouTube URL passed to web crawler — use transcript API instead: %s", url)
        return None
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": _BROWSER_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=CRAWL_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for el in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            el.decompose()
        # Prefer article/main content when present
        main = soup.find("article") or soup.find("main") or soup.body
        text = " ".join((main or soup).stripped_strings)
        return text if len(text.strip()) >= 50 else None
    except Exception as exc:
        logger.error("Error crawling %s: %s", url, exc)
        return None


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
            headers={"User-Agent": _BROWSER_UA},
            timeout=CRAWL_TIMEOUT_SECONDS,
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
        logger.warning("YouTube oEmbed failed: %s", exc)
        return None


def _youtube_gemini_fallback(video_url: str, page_text: str | None) -> str | None:
    """Last resort: use Gemini on scraped title/description."""
    if not page_text:
        return None
    try:
        ensure_gemini_configured()
        prompt = (
            f"The following is metadata from a YouTube video ({video_url}). "
            "Expand it into detailed educational notes covering the likely topics, "
            "key concepts, and learning points. Be thorough.\n\n"
            f"{page_text[:8000]}"
        )
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content(prompt)
        return (resp.text or "").strip() or None
    except Exception as exc:
        logger.error("YouTube Gemini fallback failed: %s", exc)
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
        language_sets = [
            ["en", "en-US", "en-GB"],
            ["en"],
            ["hi", "es", "fr", "de", "pt", "ja", "ko", "ar"],
        ]
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
        logger.error("YouTube transcript error for %s: %s", vid_id, exc)

    # Fallback: oEmbed metadata → Gemini educational notes (works on cloud when captions API is blocked)
    meta = _fetch_youtube_oembed(watch_url)
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


def ingest_text_source(source: str, source_type: str) -> tuple[str | None, str, str]:
    """
    Ingest URL or raw text. Returns (text, error_detail, effective_source_type).
    Auto-detects YouTube URLs even when submitted as 'web'.
    """
    source = (source or "").strip()
    stype = (source_type or "").strip().lower()

    if is_youtube_url(source):
        text, detail = get_youtube_transcript(source)
        return text, detail, "youtube"

    if stype == "youtube":
        text, detail = get_youtube_transcript(source)
        return text, detail, "youtube"

    if stype == "text":
        return source, "", "text"

    if stype == "web":
        if not source.startswith(("http://", "https://")):
            source = "https://" + source
        text = crawl_url(source)
        if not text:
            return None, (
                "Could not extract readable text. For YouTube links use the YouTube tab. "
                "For blocked sites, paste content as Text."
            ), "web"
        return text, "", "web"

    return None, f"Unknown source type: {stype}", stype


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