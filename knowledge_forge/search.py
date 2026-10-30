"""Search tab backend: filtered hybrid search with highlighted snippets, grouped by page."""
from __future__ import annotations

import re

from knowledge_forge.knowledge_base import KnowledgeBase, tokenize

_STOP = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this to was were what when "
    "where which who why with".split()
)


def query_terms(query: str) -> list[str]:
    terms = [t for t in tokenize(query) if t not in _STOP and len(t) > 1]
    return list(dict.fromkeys(terms)) or tokenize(query)[:5]


def highlight_spans(text: str, terms: list[str]) -> list[list[int]]:
    """Non-overlapping ``[start, end]`` spans of whole-word (prefix) matches of ``terms``."""
    if not terms or not text:
        return []
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\w*",
                         re.I)
    return [[m.start(), m.end()] for m in pattern.finditer(text)][:200]


def make_snippet(text: str, terms: list[str], width: int = 280) -> tuple[str, list[list[int]]]:
    """A window of ``text`` around the densest cluster of matches, with spans relative to it."""
    spans = highlight_spans(text, terms)
    if len(text) <= width:
        return text, spans
    best, best_count = 0, -1
    for s, _e in spans:
        start = max(0, s - width // 4)
        count = sum(1 for a, b in spans if a >= start and b <= start + width)
        if count > best_count:
            best, best_count = start, count
    start = best
    if start:
        sp = text.find(" ", start)
        start = sp + 1 if 0 <= sp < start + 30 else start
    end = min(len(text), start + width)
    if end < len(text):
        sp = text.rfind(" ", start, end)
        end = sp if sp > start + width // 2 else end
    prefix, suffix = ("… " if start else ""), (" …" if end < len(text) else "")
    shift = len(prefix) - start
    rel = [[a + shift, b + shift] for a, b in spans if a >= start and b <= end]
    return prefix + text[start:end] + suffix, rel


def filter_sources(kb: KnowledgeBase, sources=None, sites=None, types=None, groups=None) -> set[str] | None:
    """Sources matching every given filter (``None`` = no filtering at all)."""
    if not any((sources, sites, types, groups)):
        return None
    out = set()
    for s in kb.list_sources():
        if sources and s["source"] not in sources:
            continue
        if sites and s.get("site") not in sites:
            continue
        if types and s.get("type") not in types:
            continue
        if groups and s.get("group") not in groups:
            continue
        out.add(s["source"])
    return out


def search(kb: KnowledgeBase, query: str, *, sources=None, sites=None, types=None, groups=None,
           k: int = 20, per_page: int = 3, group_by_page: bool = True) -> dict:
    allowed = filter_sources(kb, sources, sites, types, groups)
    if allowed is not None and not allowed:
        return {"query": query, "terms": query_terms(query), "total": 0, "results": [], "pages": []}
    meta = {s["source"]: s for s in kb.list_sources()}
    terms = query_terms(query)
    hits = kb.scored_candidates(query, allowed, limit=max(k * 3, 30))
    results = []
    for h in hits[: k * 3]:
        snippet, spans = make_snippet(h["text"], terms)
        m = meta.get(h["source"], {})
        results.append({
            "id": h["id"], "source": h["source"], "source_type": h.get("source_type", "text"),
            "title": m.get("title") or (h.get("metadata") or {}).get("title") or h["source"],
            "site": m.get("site", ""), "group": m.get("group", ""),
            "score": h["score"], "semantic": h["semantic"], "keyword": h["keyword"],
            "snippet": snippet, "highlights": spans, "text": h["text"],
            "matched_terms": sorted({t for t in terms if re.search(r"\b" + re.escape(t), h["text"], re.I)}),
        })
    pages: list[dict] = []
    if group_by_page:
        by_src: dict[str, dict] = {}
        for r in results:
            page = by_src.get(r["source"])
            if page is None:
                if len(by_src) >= k:
                    continue
                page = by_src[r["source"]] = {
                    "source": r["source"], "title": r["title"], "site": r["site"], "group": r["group"],
                    "source_type": r["source_type"], "score": r["score"], "hits": [],
                }
                pages.append(page)
            if len(page["hits"]) < per_page:
                page["hits"].append(r)
    return {"query": query, "terms": terms, "total": len(results), "results": results[:k], "pages": pages}
