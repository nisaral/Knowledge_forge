"""Generation helpers and robust parsing/validation of model output."""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator

from knowledge_forge import gemini
from knowledge_forge.config import get_settings
from knowledge_forge.errors import MISSING_KEY_MESSAGE, ConfigurationError, LLMError

logger = logging.getLogger(__name__)


def ensure_gemini_configured() -> None:
    """Fail fast with a clear message when no API key is configured."""
    if not get_settings().gemini_api_key:
        raise ConfigurationError(MISSING_KEY_MESSAGE)


def generate_content(prompt: str, system: str | None = None, json_mode: bool = False) -> str:
    """Generate text with retries/backoff (see :mod:`knowledge_forge.gemini`)."""
    ensure_gemini_configured()
    return gemini.generate(prompt, system=system, json_mode=json_mode)


def stream_content(prompt: str, system: str | None = None) -> Iterator[str]:
    ensure_gemini_configured()
    return gemini.generate_stream(prompt, system=system)


# ─── Output cleaning ───────────────────────────────────────────────────────
_FENCE = r"```{lang}\s*\n?(.*?)```"


def clean_llm_output(raw: str, output_type: str) -> str:
    """Extract the fenced block of ``output_type`` (mermaid/json) from model output."""
    raw = raw or ""
    if output_type == "mermaid":
        m = re.search(_FENCE.format(lang="mermaid"), raw, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else raw.strip().strip("`").strip()
    if output_type == "json":
        m = re.search(_FENCE.format(lang="(?:json)?"), raw, re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip()[:1] in "[{":
            return m.group(1).strip()
        m = re.search(r"(\[.*\]|\{.*\})", raw, re.DOTALL)
        return m.group(0) if m else ""
    return raw


def _loads_lenient(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Common model slips: trailing commas, smart quotes.
    fixed = re.sub(r",\s*([\]}])", r"\1", text).replace("\u201c", '"').replace("\u201d", '"')
    return json.loads(fixed)


def parse_json_output(raw: str) -> dict | list:
    """Parse JSON from model output, raising :class:`LLMError` when impossible."""
    cleaned = clean_llm_output(raw, "json")
    if not cleaned:
        raise LLMError("The model did not return JSON.")
    try:
        return _loads_lenient(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMError("The model returned invalid JSON.") from exc


def _unwrap_list(data, *keys: str) -> list:
    """Accept either a bare list or an object wrapping it (``{"cards": [...]}``)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                return data[key]
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return []


def _text(value, limit: int = 2000) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()[:limit]


def validate_flashcards(data) -> list[dict]:
    cards = []
    for item in _unwrap_list(data, "flashcards", "cards"):
        if not isinstance(item, dict):
            continue
        q = _text(item.get("question") or item.get("front") or item.get("q"))
        a = _text(item.get("answer") or item.get("back") or item.get("a"))
        if q and a:
            cards.append({"question": q, "answer": a})
    return cards


def validate_storyboard(data) -> list[dict]:
    panels = []
    for item in _unwrap_list(data, "panels", "storyboard"):
        if isinstance(item, dict):
            title = _text(item.get("title"), 200)
            scene = _text(item.get("scene") or item.get("description"))
            if title or scene:
                panels.append({"title": title or "Scene", "scene": scene})
    return panels


_LETTERS = "ABCDEFGH"


def validate_quiz(data) -> list[dict]:
    """Normalise quiz questions to ``{question, options[], answer (index), explanation}``."""
    out = []
    for item in _unwrap_list(data, "questions", "quiz"):
        if not isinstance(item, dict):
            continue
        question = _text(item.get("question"), 1000)
        options = item.get("options") or item.get("choices")
        if isinstance(options, dict):
            options = [options[k] for k in sorted(options)]
        if not question or not isinstance(options, list):
            continue
        options = [re.sub(r"^\s*[A-Ha-h][).:]\s+", "", _text(o, 500)) for o in options if _text(o)]
        if not 2 <= len(options) <= len(_LETTERS):
            continue
        answer = item.get("answer", item.get("answer_index", item.get("correct")))
        idx = None
        if isinstance(answer, bool):
            idx = None
        elif isinstance(answer, int):
            idx = answer
        elif isinstance(answer, str):
            a = answer.strip()
            if a.isdigit():
                idx = int(a)
            elif len(a) == 1 and a.upper() in _LETTERS:
                idx = _LETTERS.index(a.upper())
            else:
                matches = [i for i, o in enumerate(options) if o.lower() == a.lower()]
                idx = matches[0] if matches else None
        if idx is None or not 0 <= idx < len(options):
            continue
        out.append({
            "question": question,
            "options": options,
            "answer": idx,
            "explanation": _text(item.get("explanation"), 1000),
        })
    return out


def parse_rerank(raw: str, n_candidates: int, k: int) -> list[int]:
    """Parse a re-rank response into unique zero-based indices (at most ``k``)."""
    try:
        data = _unwrap_list(parse_json_output(raw), "ranking", "order", "passages")
    except LLMError:
        data = [int(x) for x in re.findall(r"\d+", raw or "")]
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


# ─── Mermaid ───────────────────────────────────────────────────────────────
_NODE = re.compile(r"(\w+)(\[[^\]]+?\]|\([^)]+?\)|\{[^}]+?\})")
_MERMAID_HEADERS = ("graph", "flowchart", "mindmap")


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


def normalize_mermaid(raw: str, kind: str) -> str:
    """Clean model output into a Mermaid diagram with a valid header.

    Strips fences, ``%%{init}%%`` directives and ``click``/``style`` lines (which could
    load URLs or scripts), and adds a ``graph TD``/``mindmap`` header when missing.
    """
    code = clean_llm_output(raw, "mermaid")
    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        if re.match(r"(click|style|classDef|class|linkStyle)\b", stripped):
            continue
        lines.append(line.rstrip())
    if not lines:
        raise LLMError("The model did not return a diagram. Please try again.")
    first = lines[0].strip().lower()
    if not first.startswith(_MERMAID_HEADERS):
        lines.insert(0, "mindmap" if kind == "mindmap" else "graph TD")
    code = "\n".join(lines)
    return code if kind == "mindmap" else sanitize_mermaid(code)


def clean_label(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[\"*`#\n]", "", text or "").strip()
    return text[:max_len] or "Topic"
