"""LLM generation helpers and output parsing."""
import json
import logging
import re
import time

import google.generativeai as genai

from config import GEN_MODEL, LLM_MAX_RETRIES
from services.embeddings import ensure_gemini_configured

logger = logging.getLogger(__name__)


def generate_content(prompt: str) -> str:
    """Call Gemini with exponential-backoff retry."""
    ensure_gemini_configured()
    model = genai.GenerativeModel(GEN_MODEL)
    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            logger.warning("LLM retry %d/%d: %s", attempt + 1, LLM_MAX_RETRIES, exc)
            time.sleep(wait)
    raise RuntimeError(f"LLM generation failed after {LLM_MAX_RETRIES} attempts: {last_err}")


def clean_llm_output(raw: str, output_type: str) -> str:
    if output_type == "mermaid":
        m = re.search(r"```mermaid(.*?)```", raw, re.DOTALL)
        return m.group(1).strip() if m else raw.strip()
    if output_type == "json":
        m = re.search(r"```json(.*?)```", raw, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        return m.group(0) if m else "{}"
    return raw


def sanitize_mermaid(code: str) -> str:
    sanitized = []
    node_pat = re.compile(r"(\w+)(\[.+?\]|\(.+?\)|{.+?})")
    for line in code.split("\n"):
        m = node_pat.search(line)
        if m:
            before = line[:m.start(2)]
            text = m.group(2)[1:-1]
            if re.search(r"[()\[\]{}:+]", text) and not (text.startswith('"') and text.endswith('"')):
                text = text.replace('"', "&quot;")
                sanitized.append(f'{before}["{text}"]')
            else:
                sanitized.append(line)
        else:
            sanitized.append(line)
    return "\n".join(sanitized)


def parse_json_output(raw: str) -> dict | list:
    return json.loads(clean_llm_output(raw, "json"))