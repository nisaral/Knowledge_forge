"""Shared plumbing for the Create hub and the knowledge graph.

* :func:`gather` picks the passages a generator works from (topic search or whole scope).
* :func:`batches` splits them so large knowledge bases are processed map-reduce style.
* :func:`llm_json` asks for JSON, validates it, repairs once, and returns ``None`` (never
  raises on a bad format) so each caller can fall back to a deterministic heuristic.
* Citations are global passage numbers ``n`` (1-based) that map back to chunks.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable

from knowledge_forge.errors import ConfigurationError, KnowledgeForgeError, LLMError, ValidationError
from knowledge_forge.llm import generate_content, parse_json_output
from knowledge_forge.prompts import SYSTEM_INSTRUCTION, neutralize, repair_json_prompt

logger = logging.getLogger(__name__)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
BATCH_CHARS = 12_000
MAX_PASSAGES = 400


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text or "") if len(s.strip()) > 2]


def scope_sources(kb, sources=None, groups=None) -> set[str] | None:
    """Sources selected by explicit names and/or crawl groups (``None`` = whole KB)."""
    if not sources and not groups:
        return None
    out = set()
    for s in kb.list_sources():
        if (sources and s["source"] in sources) or (groups and s.get("group") in groups):
            out.add(s["source"])
    if not out:
        raise ValidationError("No sources match that selection.")
    return out


def gather(kb, topic: str | None = None, sources: set[str] | None = None, limit: int = 60) -> list[dict]:
    """Passages for a generator: topic-ranked when a topic is given, else spread over the scope.

    Without a topic, chunks are taken round-robin across sources (document order within
    each), so every source is represented before any one dominates.
    """
    if topic:
        hits = kb.scored_candidates(topic, sources, limit=limit)
        return [h for h in hits if h.get("text")]
    chunks = [c for c in kb.list_chunks() if sources is None or c["source"] in sources]
    by_src: dict[str, list[dict]] = {}
    for c in chunks:
        by_src.setdefault(c["source"], []).append(c)
    out: list[dict] = []
    queues = list(by_src.values())
    i = 0
    while len(out) < limit and any(i < len(q) for q in queues):
        for q in queues:
            if i < len(q) and len(out) < limit:
                out.append(q[i])
        i += 1
    return out


def batches(passages: list[dict], budget: int = BATCH_CHARS) -> list[list[tuple[int, dict]]]:
    """Split numbered passages ``[(n, chunk)]`` into batches of at most ``budget`` characters."""
    out, cur, used = [], [], 0
    for n, p in enumerate(passages, start=1):
        size = len(p.get("text", "")) + 60
        if cur and used + size > budget:
            out.append(cur)
            cur, used = [], 0
        cur.append((n, p))
        used += size
    if cur:
        out.append(cur)
    return out


def numbered_context(batch: list[tuple[int, dict]]) -> str:
    """Delimited passages that keep their global number, so citations survive map-reduce."""
    blocks = []
    for n, p in batch:
        source = neutralize(str(p.get("source", "unknown"))).replace('"', "'")[:200]
        blocks.append(f'<passage n="{n}" source="{source}">\n{neutralize(p.get("text", ""))}\n</passage>')
    return "\n\n".join(blocks)


def citation(n: int, passages: list[dict], snippet_chars: int = 240) -> dict | None:
    if not isinstance(n, int) or not 1 <= n <= len(passages):
        return None
    p = passages[n - 1]
    return {"n": n, "id": p.get("id"), "source": p.get("source", ""), "snippet": (p.get("text") or "")[:snippet_chars]}


def clean_citations(raw, n_passages: int, limit: int = 6) -> list[int]:
    """Valid, unique passage numbers from a model's ``citations`` field (ints, "3", "[3]")."""
    if isinstance(raw, (int, str)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for v in raw:
        if isinstance(v, bool):
            continue
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            v = int(m.group()) if m else None
        if isinstance(v, (int, float)) and 1 <= int(v) <= n_passages and int(v) not in out:
            out.append(int(v))
    return out[:limit]


def text_field(value, limit: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value) if isinstance(value, (int, float)) else ""
    return " ".join(value.split())[:limit]


def llm_json(prompt: str, validate: Callable, shape: str):
    """Generate + validate JSON with one repair attempt. ``None`` when the output stays invalid.

    Configuration errors (no key) propagate; transient LLM errors return ``None`` so the
    caller can fall back to its heuristic and say so.
    """
    try:
        raw = generate_content(prompt, system=SYSTEM_INSTRUCTION, json_mode=True)
    except ConfigurationError:
        raise
    except KnowledgeForgeError as exc:
        logger.warning("LLM call failed (%s); falling back", type(exc).__name__)
        return None
    try:
        result = validate(parse_json_output(raw))
        if result:
            return result
    except (LLMError, ValueError, TypeError):
        pass
    try:
        repaired = generate_content(repair_json_prompt(raw, shape), json_mode=True)
        return validate(parse_json_output(repaired)) or None
    except ConfigurationError:
        raise
    except (KnowledgeForgeError, ValueError, TypeError):
        return None


def llm_text(prompt: str) -> str | None:
    """Plain-text generation that returns ``None`` on transient failure (caller falls back)."""
    try:
        return (generate_content(prompt, system=SYSTEM_INSTRUCTION) or "").strip() or None
    except ConfigurationError:
        raise
    except KnowledgeForgeError as exc:
        logger.warning("LLM call failed (%s); falling back", type(exc).__name__)
        return None
