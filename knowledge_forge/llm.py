"""Gemini generation helpers and robust parsing of model output."""
from __future__ import annotations

import json
import logging
import re
import time

import google.generativeai as genai

from knowledge_forge.config import get_settings
from knowledge_forge.errors import MISSING_KEY_MESSAGE, ConfigurationError, LLMError

logger = logging.getLogger(__name__)

_configured_key: str | None = None


def ensure_gemini_configured() -> None:
    """Configure the Gemini SDK once per API key, or raise a clear error."""
    global _configured_key
    key = get_settings().gemini_api_key
    if not key:
        raise ConfigurationError(MISSING_KEY_MESSAGE)
    if key != _configured_key:
        genai.configure(api_key=key)
        _configured_key = key


def _backoff(attempt: int) -> float:
    return min(8.0, 2.0 ** attempt)


def generate_content(prompt: str) -> str:
    """Call Gemini with exponential-backoff retry."""
    ensure_gemini_configured()
    settings = get_settings()
    model = genai.GenerativeModel(settings.gen_model)
    last_err: Exception | None = None
    for attempt in range(settings.llm_max_retries):
        try:
            resp = model.generate_content(prompt)
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                return text
            raise LLMError("The model returned an empty response.")
        except Exception as exc:  # SDK raises many exception types
            last_err = exc
            logger.warning("LLM attempt %d/%d failed: %s", attempt + 1, settings.llm_max_retries, exc)
            if attempt + 1 < settings.llm_max_retries:
                time.sleep(_backoff(attempt))
    raise LLMError(f"The language model failed after {settings.llm_max_retries} attempts: {last_err}")


_FENCE = r"```{lang}\s*\n?(.*?)```"


def clean_llm_output(raw: str, output_type: str) -> str:
    """Extract the fenced block of ``output_type`` (mermaid/json) from model output."""
    raw = raw or ""
    if output_type == "mermaid":
        m = re.search(_FENCE.format(lang="mermaid"), raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else raw.strip().strip("`").strip()
    if output_type == "json":
        m = re.search(_FENCE.format(lang="json"), raw, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        return m.group(0) if m else ""
    return raw


def parse_json_output(raw: str) -> dict | list:
    """Parse JSON from model output, raising :class:`LLMError` when impossible."""
    cleaned = clean_llm_output(raw, "json")
    if not cleaned:
        raise LLMError("The model did not return JSON.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Common model slip: trailing commas.
        try:
            return json.loads(re.sub(r",\s*([\]}])", r"\1", cleaned))
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned invalid JSON.") from exc


def parse_rerank(raw: str, n_candidates: int, k: int) -> list[int]:
    """Parse a re-rank response into unique zero-based indices (at most ``k``)."""
    try:
        data = parse_json_output(raw)
    except LLMError:
        data = [int(x) for x in re.findall(r"\d+", raw or "")]
    if not isinstance(data, list):
        return []
    out: list[int] = []
    for item in data:
        try:
            n = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= n_candidates and (n - 1) not in out:
            out.append(n - 1)
        if len(out) >= k:
            break
    return out


_NODE = re.compile(r"(\w+)(\[[^\]]+?\]|\([^)]+?\)|\{[^}]+?\})")


def sanitize_mermaid(code: str) -> str:
    """Quote node labels that contain characters Mermaid would choke on."""
    out = []
    for line in (code or "").split("\n"):
        m = _NODE.search(line)
        if m:
            label = m.group(2)[1:-1]
            needs_quote = re.search(r"[()\[\]{}:+;]", label)
            quoted = label.startswith('"') and label.endswith('"')
            if needs_quote and not quoted:
                safe = label.replace('"', "#quot;")
                line = f'{line[:m.start(2)]}["{safe}"]{line[m.end(2):]}'
        out.append(line)
    return "\n".join(out)


def clean_label(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[\"*`#\n]", "", text or "").strip()
    return text[:max_len] or "Topic"
