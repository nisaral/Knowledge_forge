"""Search tab: ranking, filters, snippets/highlights and the /api/search endpoint."""
from knowledge_forge import search
from knowledge_forge.web import create_app

DOCS = {
    "https://bio.example/photo": ("Photosynthesis in chloroplasts converts light energy into glucose. "
                                  "Chlorophyll absorbs red and blue light.", "web", "site:bio.example", "bio.example"),
    "https://bio.example/cell": ("Mitochondria are the powerhouse of the cell and produce ATP through respiration.",
                                 "web", "site:bio.example", "bio.example"),
    "notes.pdf": ("The French Revolution began in 1789 with the storming of the Bastille.", "pdf", None, None),
    "Note: physics": ("Newton's second law states force equals mass times acceleration.", "text", None, None),
}


def load(kb):
    for src, (text, typ, group, site) in DOCS.items():
        meta = {"group": group, "site": site, "title": src.rsplit("/", 1)[-1].title()} if group else {}
        kb.sync_source([text], src, typ, meta)


def test_ranking_prefers_matching_page(kb):
    load(kb)
    out = search.search(kb, "chlorophyll light energy", k=5)
    assert out["pages"][0]["source"] == "https://bio.example/photo"
    top = out["results"][0]
    assert top["keyword"] == 1.0 and top["score"] >= out["results"][-1]["score"]
    assert {"chlorophyll", "light", "energy"} <= set(top["matched_terms"])


def test_highlight_spans_and_snippet():
    text = "alpha " * 100 + "photosynthesis makes glucose " + "beta " * 100
    snip, spans = search.make_snippet(text, ["photosynthesis", "glucose"], width=120)
    assert len(snip) < 140 and snip.startswith("… ") and snip.endswith(" …")
    assert [snip[a:b] for a, b in spans] == ["photosynthesis", "glucose"]
    assert search.highlight_spans("Photosynthetic photo", ["photo"]) == [[0, 14], [15, 20]]
    assert search.query_terms("What is the Bastille?") == ["bastille"]


def test_filters(kb):
    load(kb)
    only_pdf = search.search(kb, "revolution light force", types={"pdf"})
    assert {r["source"] for r in only_pdf["results"]} == {"notes.pdf"}
    site = search.search(kb, "energy cell", sites={"bio.example"})
    assert all(r["site"] == "bio.example" for r in site["results"]) and site["results"]
    grp = search.search(kb, "force", groups={"site:bio.example"})
    assert all(r["group"] == "site:bio.example" for r in grp["results"])
    none = search.search(kb, "force", types={"youtube"})
    assert none["results"] == [] and none["pages"] == []
    one = search.search(kb, "force mass", sources={"Note: physics"})
    assert [p["source"] for p in one["pages"]] == ["Note: physics"]


def test_grouping_caps_hits_per_page(kb):
    kb.sync_source([f"Glucose fact number {i} about glucose and energy." for i in range(6)], "big", "text")
    kb.sync_source(["Glucose is a sugar."], "small", "text")
    out = search.search(kb, "glucose", per_page=2)
    big = next(p for p in out["pages"] if p["source"] == "big")
    assert len(big["hits"]) == 2


def test_search_api_validation(kb):
    c = create_app(kb).test_client()
    assert c.post("/api/search", json={"query": "x"}).status_code == 422  # empty KB
    load(kb)
    assert c.post("/api/search", json={}).status_code == 400
    assert c.post("/api/search", json={"query": "x", "k": 0}).status_code == 400
    assert c.post("/api/search", json={"query": "x", "types": [1]}).status_code == 400
    assert c.post("/api/search", data="nope").status_code == 400
    r = c.post("/api/search", json={"query": "Bastille 1789", "types": "pdf", "k": 3})
    body = r.get_json()
    assert r.status_code == 200 and body["pages"][0]["source"] == "notes.pdf"
    hit = body["pages"][0]["hits"][0]
    assert {hit["snippet"][a:b].lower() for a, b in hit["highlights"]} == {"bastille", "1789"}
