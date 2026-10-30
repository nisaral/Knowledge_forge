"""Knowledge graph and the views built on it (summaries, concept map, glossary, timeline,
"connect the dots" paths and related-source suggestions).

Extraction is map-reduce: each batch of numbered passages is sent to the model for
entities + relations (map); merging, validation and importance scoring happen in Python
(reduce). When the model fails, a co-occurrence heuristic over capitalised phrases is used
and the result is flagged ``fallback``.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict, deque

import numpy as np

from knowledge_forge import studio
from knowledge_forge.cache import get_kb_version
from knowledge_forge.knowledge_base import tokenize

ENTITY_TYPES = ("Person", "Organization", "Location", "Concept", "Event", "Technology", "Date", "Work", "Other")
_TYPE_ALIASES = {"org": "Organization", "organisation": "Organization", "company": "Organization",
                 "place": "Location", "country": "Location", "city": "Location", "people": "Person",
                 "idea": "Concept", "theory": "Concept", "tech": "Technology", "tool": "Technology",
                 "time": "Date", "year": "Date", "book": "Work", "product": "Technology"}
GRAPH_SHAPE = ('{"entities": [{"name": str, "type": str, "description": str, "aliases": [str], '
               '"citations": [int]}], "relations": [{"source": str, "target": str, "label": str, '
               '"citations": [int]}]}')
MAX_NODES = 150
_cache: dict = {}
_cache_lock = threading.Lock()


def norm_name(name: str) -> str:
    s = re.sub(r"[^\w\s]", " ", (name or "").lower())
    s = re.sub(r"^(the|a|an)\s+", "", " ".join(s.split()))
    return s[:-1] if len(s) > 4 and s.endswith("s") and not s.endswith("ss") else s


def norm_type(value) -> str:
    v = str(value or "").strip()
    for t in ENTITY_TYPES:
        if v.lower() == t.lower():
            return t
    return _TYPE_ALIASES.get(v.lower(), "Other")


def validate_extraction(data, n_passages: int) -> dict | None:
    """Schema-check one batch of model output. Drops malformed items; ``None`` if nothing usable."""
    if not isinstance(data, dict):
        return None
    entities = []
    for e in data.get("entities") or []:
        if not isinstance(e, dict):
            continue
        name = studio.text_field(e.get("name"), 80)
        if len(name) < 2 or not norm_name(name):
            continue
        aliases = e.get("aliases") if isinstance(e.get("aliases"), list) else []
        entities.append({
            "name": name, "type": norm_type(e.get("type")),
            "description": studio.text_field(e.get("description"), 300),
            "aliases": [studio.text_field(a, 80) for a in aliases if isinstance(a, str) and a.strip()][:5],
            "citations": studio.clean_citations(e.get("citations"), n_passages, 8),
        })
    relations = []
    for r in data.get("relations") or []:
        if not isinstance(r, dict):
            continue
        s, t = studio.text_field(r.get("source"), 80), studio.text_field(r.get("target"), 80)
        if s and t and norm_name(s) != norm_name(t):
            relations.append({"source": s, "target": t,
                              "label": studio.text_field(r.get("label") or r.get("relation"), 60) or "related to",
                              "citations": studio.clean_citations(r.get("citations"), n_passages, 6)})
    if not entities:
        return None
    return {"entities": entities, "relations": relations}


def extraction_prompt(context: str, max_entities: int) -> str:
    return (
        f"Build a knowledge graph from the numbered passages. Extract at most {max_entities} important entities "
        f"(types: {', '.join(ENTITY_TYPES)}) and the relations between them stated in the text. Use the most "
        "complete name for each entity and list other spellings as aliases. Return ONLY JSON: "
        '{"entities": [{"name", "type", "description": one sentence from the text, "aliases": [], '
        '"citations": [passage numbers]}], "relations": [{"source": entity name, "target": entity name, '
        '"label": short verb phrase, "citations": [passage numbers]}]}.\n\n'
        f"Passages:\n{context}"
    )


_CAP_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9\-]+(?:\s+(?:of|the|and|de|von)?\s*[A-Z][a-zA-Z0-9\-]+){0,3})\b")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_SKIP = frozenset("The This That These Those It In On At For From With When While However Also Although "
                  "After Before During Many Some Most Each Every One Two Three Its Their They There Here "
                  "What Which Who Why How If Then Thus But And Or So As By To Of A An We I You He She".split())


def heuristic_extraction(batch: list[tuple[int, dict]]) -> dict:
    """Capitalised phrases as entities, sentence co-occurrence as relations."""
    entities: dict[str, dict] = {}
    pairs: Counter = Counter()
    pair_cites: dict = defaultdict(set)
    for n, p in batch:
        for sent in studio.sentences(p["text"]) or [p["text"]]:
            found = []
            for m in _CAP_RE.finditer(sent):
                name = m.group().strip()
                first = name.split()[0]
                if first in _SKIP:
                    name = " ".join(name.split()[1:])
                if len(name) < 3 or name in _SKIP:
                    continue
                key = norm_name(name)
                e = entities.setdefault(key, {"name": name, "type": "Other", "description": sent[:200],
                                              "aliases": [], "citations": []})
                if n not in e["citations"]:
                    e["citations"].append(n)
                found.append(key)
            for y in _YEAR_RE.findall(sent):
                e = entities.setdefault(y, {"name": y, "type": "Date", "description": sent[:200], "aliases": [],
                                            "citations": []})
                if n not in e["citations"]:
                    e["citations"].append(n)
                found.append(y)
            uniq = list(dict.fromkeys(found))
            for i, a in enumerate(uniq):
                for b in uniq[i + 1:i + 4]:
                    pairs[(a, b)] += 1
                    pair_cites[(a, b)].add(n)
    relations = [{"source": entities[a]["name"], "target": entities[b]["name"], "label": "mentioned with",
                  "citations": sorted(pair_cites[(a, b)])[:6]} for (a, b), _ in pairs.most_common(200)]
    return {"entities": list(entities.values()), "relations": relations}


def merge(extractions: list[dict]) -> tuple[dict, list[dict]]:
    """Reduce step: merge entities by normalised name and aliases; resolve relation endpoints."""
    nodes: dict[str, dict] = {}
    alias_to: dict[str, str] = {}
    for ex in extractions:
        for e in ex["entities"]:
            keys = [norm_name(e["name"])] + [norm_name(a) for a in e["aliases"]]
            target = next((alias_to[k] for k in keys if k in alias_to), None)
            if target is None:
                target = keys[0]
                nodes[target] = {"id": target, "name": e["name"], "type": e["type"], "description": e["description"],
                                 "aliases": [], "citations": [], "mentions": 0}
            node = nodes[target]
            node["mentions"] += max(1, len(e["citations"]))
            if node["type"] == "Other" and e["type"] != "Other":
                node["type"] = e["type"]
            if len(e["description"]) > len(node["description"]) and e["type"] != "Other":
                node["description"] = e["description"]
            if len(e["name"]) > len(node["name"]):
                node["aliases"].append(node["name"])
                node["name"] = e["name"]
            for a in [e["name"], *e["aliases"]]:
                if a != node["name"] and a not in node["aliases"]:
                    node["aliases"].append(a)
            for c in e["citations"]:
                if c not in node["citations"]:
                    node["citations"].append(c)
            for k in keys:
                alias_to.setdefault(k, target)
    edges: dict[tuple, dict] = {}
    for ex in extractions:
        for r in ex["relations"]:
            a, b = alias_to.get(norm_name(r["source"])), alias_to.get(norm_name(r["target"]))
            if not a or not b or a == b:
                continue
            key = (a, b, r["label"].lower())
            edge = edges.setdefault(key, {"source": a, "target": b, "label": r["label"], "citations": [], "weight": 0})
            edge["weight"] += 1
            edge["citations"] = sorted(set(edge["citations"]) | set(r["citations"]))[:8]
    return nodes, list(edges.values())


def score_and_trim(nodes: dict, edges: list[dict], max_nodes: int) -> tuple[list[dict], list[dict]]:
    degree = Counter()
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    for nid, n in nodes.items():
        n["degree"] = degree[nid]
        n["importance"] = round(math.log1p(n["mentions"]) + 0.7 * math.log1p(degree[nid]), 4)
    keep = sorted(nodes.values(), key=lambda n: -n["importance"])[:max_nodes]
    top = max((n["importance"] for n in keep), default=1) or 1
    for n in keep:
        n["importance"] = round(n["importance"] / top, 4)
    ids = {n["id"] for n in keep}
    return keep, [e for e in edges if e["source"] in ids and e["target"] in ids]


def build_graph(kb, *, sources=None, topic: str | None = None, max_nodes: int = 80, use_cache: bool = True):
    key = (get_kb_version(), kb.count(), frozenset(sources) if sources else None, topic, max_nodes)
    if use_cache:
        with _cache_lock:
            if key in _cache:
                return _cache[key]
    passages = studio.gather(kb, topic, sources, limit=60 if topic else 160)
    if not passages:
        return None
    extractions, fallback_batches = [], 0
    groups = studio.batches(passages)
    for batch in groups:
        got = studio.llm_json(extraction_prompt(studio.numbered_context(batch), 25),
                              lambda d: validate_extraction(d, len(passages)), GRAPH_SHAPE)
        if got is None:
            fallback_batches += 1
            got = heuristic_extraction(batch)
        extractions.append(got)
    nodes, edges = merge(extractions)
    nodes, edges = score_and_trim(nodes, edges, max_nodes)
    for n in nodes:
        n["sources"] = [c for c in (studio.citation(i, passages) for i in n["citations"][:5]) if c]
        n["source_names"] = sorted({c["source"] for c in n["sources"]})
    for e in edges:
        e["sources"] = [c for c in (studio.citation(i, passages) for i in e["citations"][:3]) if c]
    result = {"nodes": nodes, "edges": edges, "types": sorted({n["type"] for n in nodes}),
              "fallback": fallback_batches == len(groups), "partial_fallback": 0 < fallback_batches < len(groups),
              "batches": len(groups), "passages": len(passages)}
    with _cache_lock:
        if len(_cache) > 16:
            _cache.clear()
        _cache[key] = result
    return result


# ─── Views on the graph ─────────────────────────────────────────────────────
def concept_map(graph: dict, limit: int = 40) -> str:
    """Mermaid flowchart of the most important nodes and their labelled edges."""
    nodes = graph["nodes"][:limit]
    ids = {n["id"]: f"n{i}" for i, n in enumerate(nodes)}

    def esc(s):
        return re.sub(r'["\[\]{}()<>|#;`]', " ", s).strip()[:50] or "?"

    lines = ["graph LR"]
    lines += [f'  {ids[n["id"]]}["{esc(n["name"])}"]' for n in nodes]
    for e in graph["edges"]:
        if e["source"] in ids and e["target"] in ids:
            lines.append(f'  {ids[e["source"]]} -->|{esc(e["label"])}| {ids[e["target"]]}')
    return "\n".join(lines)


def glossary(graph: dict) -> list[dict]:
    out = [{"term": n["name"], "type": n["type"], "definition": n["description"], "aliases": n["aliases"][:3],
            "sources": n["sources"][:2]} for n in graph["nodes"] if n["description"] and n["type"] != "Date"]
    return sorted(out, key=lambda x: x["term"].lower())


_DATE_RE = re.compile(
    r"\b((?:(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(?:\d{1,2},\s+)?)?(1[0-9]{3}|20[0-9]{2})(?:s)?)\b")


def timeline(kb, sources=None, limit: int = 60) -> list[dict]:
    """Dated events: sentences containing a year (optionally with month), sorted chronologically."""
    events, seen = [], set()
    for c in kb.list_chunks():
        if sources is not None and c["source"] not in sources:
            continue
        for sent in studio.sentences(c["text"]):
            for m in _DATE_RE.finditer(sent):
                key = (m.group(2), sent[:80])
                if key in seen or len(sent) > 400:
                    continue
                seen.add(key)
                events.append({"date": m.group(1), "year": int(m.group(2)), "text": sent,
                               "source": c["source"], "id": c["id"]})
    events.sort(key=lambda e: (e["year"], e["date"]))
    return events[:limit]


def find_paths(graph: dict, a: str, b: str, max_paths: int = 3, max_len: int = 5) -> tuple[dict, dict, list]:
    """Shortest undirected paths between the nodes best matching ``a`` and ``b``."""
    def match(q):
        nq = norm_name(q)
        best = None
        for n in graph["nodes"]:
            names = [norm_name(x) for x in [n["name"], *n["aliases"]]]
            score = 3 if nq in names else 2 if any(nq and (nq in x or x in nq) for x in names) else 0
            if score and (best is None or (score, n["importance"]) > best[0]):
                best = ((score, n["importance"]), n)
        return best[1] if best else None

    na, nb = match(a), match(b)
    if not na or not nb or na is nb:
        return na, nb, []
    adj = defaultdict(list)
    for e in graph["edges"]:
        adj[e["source"]].append((e["target"], e))
        adj[e["target"]].append((e["source"], e))
    paths, queue = [], deque([[na["id"]]])
    edge_paths = {(na["id"],): []}
    shortest = None
    while queue and len(paths) < max_paths:
        path = queue.popleft()
        if len(path) > max_len or (shortest and len(path) > shortest + 1):
            break
        if path[-1] == nb["id"]:
            paths.append({"nodes": path, "edges": edge_paths[tuple(path)]})
            shortest = shortest or len(path)
            continue
        for nxt, e in adj[path[-1]]:
            if nxt not in path:
                new = path + [nxt]
                edge_paths[tuple(new)] = edge_paths[tuple(path)] + [e]
                queue.append(new)
    return na, nb, paths


def describe_path(graph: dict, path: dict) -> str:
    names = {n["id"]: n["name"] for n in graph["nodes"]}
    parts = []
    for i, e in enumerate(path["edges"]):
        a, b = path["nodes"][i], path["nodes"][i + 1]
        if e["source"] == a:
            parts.append(f"{names[a]} {e['label']} {names[b]}")
        else:
            parts.append(f"{names[b]} {e['label']} {names[a]}")
    return "; ".join(parts) + "."


def keywords(texts: list[str], all_texts: list[str], k: int = 8) -> list[str]:
    """TF-IDF key terms of ``texts`` against the corpus ``all_texts``."""
    df = Counter()
    for t in all_texts:
        df.update(set(tokenize(t)))
    tf = Counter(w for t in texts for w in tokenize(t) if len(w) > 3 and not w.isdigit())
    n = max(1, len(all_texts))
    scored = {w: c * math.log((n + 1) / (df[w] + 1) + 1) for w, c in tf.items()}
    return [w for w, _ in sorted(scored.items(), key=lambda x: -x[1])[:k]]


def related_sources(kb, source: str, k: int = 5) -> list[dict]:
    """Sources similar to ``source``: mean-embedding cosine plus shared key terms."""
    chunks = kb.list_chunks()
    by_src: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_src[c["source"]].append(c)
    if source not in by_src:
        return []
    vecs = kb.source_vectors() if hasattr(kb, "source_vectors") else {}
    all_texts = [c["text"] for c in chunks]
    terms = {s: set(keywords([c["text"] for c in cs], all_texts, 15)) for s, cs in by_src.items()}
    base = vecs.get(source)
    out = []
    for s in by_src:
        if s == source:
            continue
        sim = 0.0
        if base is not None and vecs.get(s) is not None:
            a, b = base, vecs[s]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            sim = float(np.dot(a, b) / denom)
        shared = sorted(terms[source] & terms[s])
        score = 0.75 * max(sim, 0.0) + 0.25 * (len(shared) / 15)
        out.append({"source": s, "score": round(score, 4), "similarity": round(sim, 4), "shared_terms": shared[:8]})
    return sorted(out, key=lambda x: -x["score"])[:k]


def reset_graph_cache() -> None:
    with _cache_lock:
        _cache.clear()
