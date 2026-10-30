"""Storyboard + flashcards: schema validation, LLM path, repair, fallback, map-reduce, endpoints."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from knowledge_forge import create, studio
from knowledge_forge.errors import LLMError
from knowledge_forge.web import create_app

TEXTS = {
    "bio.txt": "Photosynthesis converts light energy into chemical energy in chloroplasts. Chlorophyll absorbs "
               "red and blue light. The Calvin cycle fixes carbon dioxide into glucose.",
    "hist.txt": "The French Revolution began in 1789 with the storming of the Bastille. Napoleon Bonaparte "
                "became First Consul in 1799 after the coup of Brumaire.",
}


def load(kb):
    for src, text in TEXTS.items():
        kb.sync_source([text], src, "text")


def fake_llm(monkeypatch, *responses):
    calls = []
    queue = list(responses)

    def gen(prompt, system=None, json_mode=False):
        calls.append(prompt)
        r = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)

    monkeypatch.setattr(studio, "generate_content", gen)
    return calls


FRAMES = {"frames": [
    {"title": "Light in", "narration": "Plants catch light.", "key_points": ["chlorophyll"], "visual": "a leaf",
     "icon": "leaf", "citations": [1, "2", "[99]"], "duration_sec": 8},
    {"title": "Sugar out", "narration": "Carbon becomes glucose.", "key_points": "Calvin cycle",
     "icon": "not-an-icon", "citations": [], "duration_sec": 999},
    "garbage", {"foo": 1},
]}


def test_validate_frames_cleans_schema():
    frames = create.validate_frames(FRAMES, n_passages=2)
    assert len(frames) == 2
    assert frames[0]["citations"] == [1, 2] and frames[0]["icon"] == "leaf"
    assert frames[1]["icon"] in create.ICONS and frames[1]["key_points"] == ["Calvin cycle"]
    assert 3 <= frames[1]["duration_sec"] <= 60


def test_storyboard_llm_path(kb, monkeypatch):
    load(kb)
    fake_llm(monkeypatch, FRAMES)
    out = create.generate_storyboard(kb, topic="photosynthesis", frames=2)
    assert not out["fallback"] and len(out["frames"]) == 2
    assert all(f["citations"] and f["sources"] for f in out["frames"])
    assert out["total_duration_sec"] == sum(f["duration_sec"] for f in out["frames"])


def test_storyboard_repairs_then_falls_back(kb, monkeypatch):
    load(kb)
    calls = fake_llm(monkeypatch, "not json", json.dumps(FRAMES))
    assert not create.generate_storyboard(kb, topic=None, frames=2)["fallback"]
    assert len(calls) == 2  # original + one repair
    fake_llm(monkeypatch, "still not json")
    out = create.generate_storyboard(kb, topic=None, frames=3)
    assert out["fallback"] and len(out["frames"]) == 2  # only 2 passages available
    assert all(f["citations"] for f in out["frames"])
    fake_llm(monkeypatch, LLMError("boom"))
    assert create.generate_storyboard(kb, topic="light", frames=2)["fallback"]


def test_map_reduce_for_large_inputs(kb, monkeypatch):
    for i in range(12):
        kb.sync_source([f"Topic {i} paragraph. " + "word " * 400], f"doc{i}.txt", "text")
    notes = {"notes": [{"point": "Condensed fact", "citations": [1]}]}
    calls = fake_llm(monkeypatch, notes)
    passages = studio.gather(kb, None, None, limit=50)
    ctx, reduced = create.map_reduce_context("x", passages, budget=3000)
    assert reduced and "Condensed fact [1]" in ctx and len(calls) > 1


def test_regenerate_frame(kb, monkeypatch):
    load(kb)
    fake_llm(monkeypatch, {"title": "New", "narration": "Fresh take.", "citations": [1]})
    out = create.regenerate_frame(kb, frames=[{"title": "A"}, {"title": "B"}], index=1, topic="light",
                                  instruction="simpler")
    assert out["frame"]["title"] == "New" and not out["fallback"]


def test_flashcards_validation_and_fallback(kb, monkeypatch):
    load(kb)
    cards = {"cards": [
        {"question": "What absorbs light?", "answer": "Chlorophyll", "difficulty": "EASY", "citations": [1]},
        {"question": "what absorbs light?", "answer": "dup"},
        {"front": "When did the Revolution begin?", "back": "1789", "tags": ["history", 5]},
        {"question": "", "answer": "x"},
    ]}
    fake_llm(monkeypatch, cards)
    out = create.generate_flashcards(kb, topic=None, count=5, difficulty="hard")
    assert not out["fallback"] and len(out["cards"]) == 2
    assert out["cards"][0]["difficulty"] == "easy" and out["cards"][1]["difficulty"] == "hard"
    assert out["cards"][1]["tags"] == ["history"] and out["cards"][0]["sources"]
    assert len({c["id"] for c in out["cards"]}) == 2
    fake_llm(monkeypatch, "[]")
    out = create.generate_flashcards(kb, topic=None, count=3, difficulty="mixed")
    assert out["fallback"] and out["cards"]
    assert all("_____" in c["question"] and c["citations"] for c in out["cards"])


def test_endpoints(kb, monkeypatch):
    app = create_app(kb).test_client()
    assert app.post("/api/flashcards", json={}).status_code == 422
    load(kb)
    fake_llm(monkeypatch, FRAMES)
    r = app.post("/api/storyboard", json={"topic": "light", "frames": 2})
    assert r.status_code == 200 and len(r.get_json()["frames"]) == 2
    assert app.post("/api/storyboard", json={"frames": "x"}).status_code == 400
    assert app.post("/api/storyboard", json={"sources": ["nope.txt"]}).status_code == 400
    r = app.post("/api/storyboard/frame", json={"frames": [{"title": "a"}], "index": 0})
    assert r.status_code == 200
    assert app.post("/api/storyboard/frame", json={"frames": "x"}).status_code == 400
    fake_llm(monkeypatch, {"cards": [{"question": "Q?", "answer": "A", "citations": [1]}]})
    r = app.post("/api/flashcards", json={"count": 3, "difficulty": "easy", "sources": ["bio.txt"]})
    assert r.status_code == 200 and r.get_json()["cards"][0]["sources"][0]["source"] == "bio.txt"
    assert app.post("/api/flashcards", json={"difficulty": "insane"}).status_code == 400
    assert app.post("/api/flashcards", json={"count": 500}).status_code == 400


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_sm2_js():
    script = Path(__file__).parent / "sm2_check.mjs"
    res = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stdout + res.stderr
