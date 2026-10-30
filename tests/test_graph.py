"""Knowledge graph: schema validation, merging, importance, fallback, paths, views and endpoints."""
import json

from knowledge_forge import graph, studio
from knowledge_forge.web import create_app

TEXTS = {
    "rev.txt": "The French Revolution began in 1789 with the storming of the Bastille in Paris. "
               "Napoleon Bonaparte became First Consul in 1799.",
    "nap.txt": "Napoleon crowned himself Emperor of the French in December 1804. He was defeated at Waterloo in 1815.",
    "bio.txt": "Photosynthesis happens in chloroplasts. Chlorophyll absorbs light energy.",
}
EXTRACTION = {
    "entities": [
        {"name": "Napoleon Bonaparte", "type": "person", "description": "French military leader.",
         "aliases": ["Napoleon"], "citations": [1, 2]},
        {"name": "Napoleon", "type": "Person", "description": "", "citations": [2]},
        {"name": "French Revolution", "type": "event", "description": "Revolution that began in 1789.",
         "citations": [1]},
        {"name": "Waterloo", "type": "place", "citations": ["2"]},
        {"name": "x"}, "junk",
    ],
    "relations": [
        {"source": "French Revolution", "target": "Napoleon", "label": "brought to power", "citations": [1]},
        {"source": "Napoleon Bonaparte", "target": "Waterloo", "label": "defeated at", "citations": [2]},
        {"source": "Napoleon", "target": "Napoleon Bonaparte", "label": "self"},
        {"source": "Nobody", "target": "Waterloo", "label": "ghost"},
    ],
}


def load(kb):
    for s, t in TEXTS.items():
        kb.sync_source([t], s, "text")


def fake_llm(monkeypatch, response):
    def gen(prompt, system=None, json_mode=False):
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)

    monkeypatch.setattr(studio, "generate_content", gen)


def test_validation_and_merge():
    v = graph.validate_extraction(EXTRACTION, n_passages=2)
    assert [e["type"] for e in v["entities"]] == ["Person", "Person", "Event", "Location"]
    assert len(v["relations"]) == 4  # alias self-loop and ghost are only dropped at merge
    assert graph.validate_extraction({"entities": []}, 2) is None
    assert graph.validate_extraction([1, 2], 2) is None
    nodes, edges = graph.merge([v])
    assert len(nodes) == 3 and nodes["napoleon bonaparte"]["mentions"] == 3
    assert "Napoleon" in nodes["napoleon bonaparte"]["aliases"]
    assert {(e["source"], e["target"]) for e in edges} == {("french revolution", "napoleon bonaparte"),
                                                           ("napoleon bonaparte", "waterloo")}
    kept, kept_edges = graph.score_and_trim(nodes, edges, 2)
    assert kept[0]["id"] == "napoleon bonaparte" and kept[0]["importance"] == 1.0 and len(kept_edges) == 1


def test_build_graph_llm_and_cache(kb, monkeypatch):
    load(kb)
    fake_llm(monkeypatch, EXTRACTION)
    g = graph.build_graph(kb)
    assert not g["fallback"] and g["nodes"][0]["name"] == "Napoleon Bonaparte"
    assert g["nodes"][0]["sources"] and g["edges"][0]["sources"]
    fake_llm(monkeypatch, RuntimeError("should be cached"))
    assert graph.build_graph(kb) is g


def test_build_graph_fallback(kb, monkeypatch):
    load(kb)
    fake_llm(monkeypatch, "nope")
    g = graph.build_graph(kb, use_cache=False)
    names = {n["name"] for n in g["nodes"]}
    assert g["fallback"] and "1789" in names and any("Napoleon" in n for n in names)
    assert g["edges"]


def test_paths_views(kb, monkeypatch):
    load(kb)
    fake_llm(monkeypatch, EXTRACTION)
    g = graph.build_graph(kb)
    a, b, paths = graph.find_paths(g, "french revolution", "Waterloo")
    assert a["name"] == "French Revolution" and b["name"] == "Waterloo"
    assert paths[0]["nodes"] == ["french revolution", "napoleon bonaparte", "waterloo"]
    assert "brought to power" in graph.describe_path(g, paths[0])
    assert graph.find_paths(g, "zzz", "Waterloo")[0] is None
    m = graph.concept_map(g)
    assert m.startswith("graph LR") and "-->|defeated at|" in m
    assert [t["term"] for t in graph.glossary(g)] == ["French Revolution", "Napoleon Bonaparte"]
    years = [e["year"] for e in graph.timeline(kb)]
    assert years == sorted(years) and {1789, 1799, 1804, 1815} <= set(years)
    assert any(e["date"] == "December 1804" for e in graph.timeline(kb))


def test_related_sources(kb):
    load(kb)
    rel = graph.related_sources(kb, "rev.txt")
    assert rel[0]["source"] == "nap.txt" and "napoleon" in rel[0]["shared_terms"]
    assert graph.related_sources(kb, "missing") == []


def test_endpoints(kb, monkeypatch):
    app = create_app(kb).test_client()
    assert app.post("/api/graph", json={}).status_code == 422
    load(kb)
    fake_llm(monkeypatch, EXTRACTION)
    r = app.post("/api/graph", json={"include_map": True})
    body = r.get_json()
    assert r.status_code == 200 and body["nodes"] and body["mermaid_code"].startswith("graph LR")
    assert app.post("/api/graph", json={"max_nodes": 1}).status_code == 400
    r = app.post("/api/connect", json={"a": "French Revolution", "b": "Waterloo"}).get_json()
    assert r["paths"][0]["names"][1] == "Napoleon Bonaparte" and r["explanation"]
    assert app.post("/api/connect", json={"a": "Mars", "b": "Waterloo"}).status_code == 404
    assert app.post("/api/connect", json={"a": "x"}).status_code == 400
    assert app.post("/api/glossary", json={}).get_json()["terms"]
    assert app.post("/api/concept-map", json={}).status_code == 200
    assert app.post("/api/timeline", json={"sources": ["nap.txt"]}).get_json()["events"][0]["year"] == 1804
    assert app.post("/api/related-sources", json={"source": "rev.txt"}).get_json()["related"]
    assert app.post("/api/related-sources", json={"source": "zz"}).status_code == 404
    fake_llm(monkeypatch, {"sources": [{"n": 1, "summary": "About the revolution."}], "overall": "History and biology.",
                           "key_topics": ["napoleon"]})
    r = app.post("/api/overview", json={}).get_json()
    assert r["overall"] == "History and biology." and r["sources"][0]["summary"] and r["sources"][0]["key_topics"]
    fake_llm(monkeypatch, "bad")
    r = app.post("/api/overview", json={"groups": ["none"]})
    assert r.status_code == 400
    r = app.post("/api/overview", json={}).get_json()
    assert r["fallback"] and r["key_topics"]
