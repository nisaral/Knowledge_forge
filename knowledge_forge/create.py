"""Create hub generators: storyboards and flashcards (LLM with validated JSON + fallbacks)."""
from __future__ import annotations

import hashlib
import re
from collections import Counter

from knowledge_forge import studio
from knowledge_forge.knowledge_base import tokenize

ICONS = ("book", "lightbulb", "gear", "chart", "globe", "flask", "users", "clock", "flag", "target", "leaf",
         "bolt", "code", "heart", "shield", "star", "question", "check")
_ICON_HINTS = {
    "flask": "experiment chemistry chemical lab molecule reaction science",
    "leaf": "plant leaf photosynthesis chlorophyll nature biology ecosystem tree",
    "bolt": "energy power electric light fast",
    "gear": "process mechanism system engine machine step workflow",
    "chart": "data growth statistics percent result measure rate",
    "globe": "world country global history geography international",
    "users": "people society team community population person",
    "clock": "time history year century era timeline",
    "code": "software code program algorithm computer model network",
    "heart": "health body medicine cell blood",
    "shield": "security safety protect risk",
    "target": "goal objective aim purpose",
    "flag": "conclusion summary takeaway",
    "lightbulb": "idea concept insight introduction",
    "question": "question why how problem",
}
DIFFICULTIES = ("easy", "medium", "hard", "mixed")
STORY_SHAPE = ('{"frames": [{"title": str, "narration": str, "key_points": [str], "visual": str, '
               '"icon": one of ' + "/".join(ICONS) + ', "citations": [int], "duration_sec": int}]}')
CARD_SHAPE = ('{"cards": [{"question": str, "answer": str, "hint": str, "difficulty": str, '
              '"tags": [str], "citations": [int]}]}')


def pick_icon(text: str) -> str:
    words = set(tokenize(text))
    best, score = "lightbulb", 0
    for icon, hints in _ICON_HINTS.items():
        s = len(words & set(hints.split()))
        if s > score:
            best, score = icon, s
    return best


def _duration(narration: str) -> int:
    return max(4, min(60, round(len(narration.split()) / 2.5) + 2))


# ─── Storyboard ─────────────────────────────────────────────────────────────
def validate_frame(item, n_passages: int) -> dict | None:
    if not isinstance(item, dict):
        return None
    title = studio.text_field(item.get("title") or item.get("heading"), 120)
    narration = studio.text_field(item.get("narration") or item.get("scene") or item.get("script"), 1500)
    if not title and not narration:
        return None
    points = item.get("key_points") or item.get("bullets") or []
    if isinstance(points, str):
        points = [points]
    points = [studio.text_field(p, 200) for p in points if isinstance(p, (str, int, float))] if isinstance(
        points, list) else []
    visual = studio.text_field(item.get("visual") or item.get("visual_description"), 400)
    icon = item.get("icon") if item.get("icon") in ICONS else pick_icon(" ".join([title, narration, visual]))
    try:
        duration = int(float(item.get("duration_sec") or item.get("duration") or 0))
    except (TypeError, ValueError):
        duration = 0
    return {
        "title": title or "Untitled frame",
        "narration": narration,
        "key_points": [p for p in points if p][:6],
        "visual": visual,
        "icon": icon,
        "citations": studio.clean_citations(item.get("citations") or item.get("sources"), n_passages),
        "duration_sec": duration if 3 <= duration <= 120 else _duration(narration),
    }


def validate_frames(data, n_passages: int, max_frames: int = 20) -> list[dict]:
    from knowledge_forge.llm import _unwrap_list

    items = _unwrap_list(data, "frames", "storyboard", "panels")
    out = [f for f in (validate_frame(i, n_passages) for i in items) if f]
    return out[:max_frames]


def heuristic_frames(passages: list[dict], count: int, topic: str | None) -> list[dict]:
    """Deterministic storyboard from the passages themselves (used when the model fails)."""
    if not passages:
        return []
    count = max(1, min(count, len(passages)))
    step = len(passages) / count
    frames = []
    for i in range(count):
        n = int(i * step) + 1
        sents = studio.sentences(passages[n - 1]["text"]) or [passages[n - 1]["text"][:300]]
        lead = sents[0]
        words = lead.split()
        title = " ".join(words[:7]).rstrip(",.;:") + ("…" if len(words) > 7 else "")
        narration = " ".join(sents[:2])[:600]
        frames.append({
            "title": title if i else (topic or title),
            "narration": narration,
            "key_points": [s[:160] for s in sents[2:5]],
            "visual": f"Illustration of: {lead[:120]}",
            "icon": pick_icon(narration),
            "citations": [n],
            "duration_sec": _duration(narration),
        })
    return frames


def storyboard_prompt(topic: str, context: str, frames: int, style: str) -> str:
    return (
        f"Create a {frames}-frame explainer storyboard about '{topic}' using ONLY the numbered passages. "
        f"Style: {style}. Frames must tell a coherent story in order (hook, build-up, key ideas, recap). "
        "Return ONLY JSON: {\"frames\": [{\"title\": short title, \"narration\": 2-4 spoken sentences, "
        "\"key_points\": [2-4 short bullets], \"visual\": one sentence describing what is on screen, "
        f"\"icon\": one of {', '.join(ICONS)}, \"citations\": [passage numbers that support the frame], "
        "\"duration_sec\": seconds to read the narration aloud}]}. Every frame must cite at least one passage.\n\n"
        f"Passages:\n{context}"
    )


def notes_prompt(topic: str, context: str) -> str:
    return (
        f"Extract the most important facts about '{topic}' from the numbered passages. Return ONLY JSON: "
        "{\"notes\": [{\"point\": one-sentence fact, \"citations\": [passage numbers]}]} with at most 12 notes.\n\n"
        f"Passages:\n{context}"
    )


def _validate_notes(n_passages: int):
    def validate(data):
        from knowledge_forge.llm import _unwrap_list

        out = []
        for item in _unwrap_list(data, "notes", "points"):
            if isinstance(item, dict):
                point = studio.text_field(item.get("point") or item.get("fact"), 400)
                if point:
                    out.append({"point": point, "citations": studio.clean_citations(item.get("citations"), n_passages)})
        return out

    return validate


def map_reduce_context(topic: str, passages: list[dict], budget: int = studio.BATCH_CHARS) -> tuple[str, bool]:
    """Context for a final prompt. Small inputs go in verbatim; large ones are first condensed
    batch by batch into cited notes (map), whose text then becomes the context (reduce)."""
    groups = studio.batches(passages, budget)
    if len(groups) <= 1:
        return studio.numbered_context(groups[0] if groups else []), False
    notes = []
    for batch in groups[:12]:
        got = studio.llm_json(notes_prompt(topic, studio.numbered_context(batch)), _validate_notes(len(passages)),
                              '{"notes": [{"point": str, "citations": [int]}]}')
        if got:
            notes += got
        else:  # keep this batch represented even if the model failed on it
            notes += [{"point": (studio.sentences(p["text"]) or [p["text"]])[0][:300], "citations": [n]}
                      for n, p in batch[:3]]
    lines = [f"- {x['point']} " + "".join(f"[{c}]" for c in x["citations"]) for x in notes]
    context = ("Condensed notes (numbers in brackets are passage citations; reuse them):\n" + "\n".join(lines))
    return context, True


def generate_storyboard(kb, *, topic: str | None, sources=None, frames: int = 6, style: str = "clear and friendly"):
    passages = studio.gather(kb, topic, sources, limit=80 if topic else 120)
    if not passages:
        return None
    subject = topic or "the material"
    context, reduced = map_reduce_context(subject, passages)
    result = studio.llm_json(storyboard_prompt(subject, context, frames, style),
                             lambda d: validate_frames(d, len(passages), frames + 2), STORY_SHAPE)
    fallback = not result
    if fallback:
        result = heuristic_frames(passages, frames, topic)
    result = result[:frames]
    for f in result:
        if not f["citations"]:
            f["citations"] = [1]
        f["sources"] = [c for c in (studio.citation(n, passages) for n in f["citations"]) if c]
    return {"title": f"Storyboard: {subject}", "topic": topic, "frames": result, "fallback": fallback,
            "map_reduce": reduced, "passages": len(passages),
            "total_duration_sec": sum(f["duration_sec"] for f in result)}


def regenerate_frame(kb, *, frames: list[dict], index: int, topic: str | None, instruction: str | None,
                     sources=None):
    query = " ".join(filter(None, [topic, frames[index].get("title"), instruction]))
    passages = studio.gather(kb, query, sources, limit=12)
    if not passages:
        return None
    outline = "\n".join(f"{i + 1}. {f.get('title', '')}" for i, f in enumerate(frames))
    prompt = (
        f"This storyboard about '{topic or 'the material'}' has these frames:\n{outline}\n\n"
        f"Rewrite frame {index + 1} only" + (f", following this instruction: {instruction}" if instruction else "")
        + ". It must fit between its neighbours. Return ONLY JSON for one frame: {\"title\", \"narration\", "
        f"\"key_points\", \"visual\", \"icon\" (one of {', '.join(ICONS)}), \"citations\", \"duration_sec\"}}.\n\n"
        f"Passages:\n{studio.numbered_context(list(enumerate(passages, start=1)))}"
    )

    def validate(data):
        if isinstance(data, dict) and "frames" not in data:
            data = [data]
        got = validate_frames(data, len(passages), 1)
        return got[0] if got else None

    frame = studio.llm_json(prompt, validate, STORY_SHAPE.replace('{"frames": [', "").rstrip("]}") + "}")
    fallback = frame is None
    if fallback:
        frame = heuristic_frames(passages, 1, frames[index].get("title"))[0]
    if not frame["citations"]:
        frame["citations"] = [1]
    frame["sources"] = [c for c in (studio.citation(n, passages) for n in frame["citations"]) if c]
    return {"frame": frame, "fallback": fallback}


# ─── Flashcards ─────────────────────────────────────────────────────────────
def validate_cards(data, n_passages: int, difficulty: str, max_cards: int) -> list[dict]:
    from knowledge_forge.llm import _unwrap_list

    cards, seen = [], set()
    for item in _unwrap_list(data, "cards", "flashcards"):
        if not isinstance(item, dict):
            continue
        q = studio.text_field(item.get("question") or item.get("front") or item.get("q"), 500)
        a = studio.text_field(item.get("answer") or item.get("back") or item.get("a"), 1000)
        key = q.lower()
        if not q or not a or key in seen:
            continue
        seen.add(key)
        d = str(item.get("difficulty") or "").lower()
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        cards.append({
            "question": q, "answer": a,
            "hint": studio.text_field(item.get("hint"), 200),
            "difficulty": d if d in ("easy", "medium", "hard") else (difficulty if difficulty != "mixed" else "medium"),
            "tags": [studio.text_field(t, 40) for t in tags if isinstance(t, str) and t.strip()][:5],
            "citations": studio.clean_citations(item.get("citations") or item.get("sources"), n_passages, 3),
        })
    return cards[:max_cards]


def _key_term(sentence: str, common: Counter) -> str | None:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", sentence)
    words = [w for w in words if w.lower() not in studio_stop]
    if not words:
        return None
    return max(words, key=lambda w: (w[0].isupper(), common[w.lower()], len(w)))


studio_stop = frozenset("about above after again against because before being below between could during "
                        "every first other their there these those through under until where which while "
                        "would should called using known often".split())


def heuristic_cards(passages: list[dict], count: int, difficulty: str) -> list[dict]:
    """Cloze cards from passage sentences (used when the model fails)."""
    common = Counter(t for p in passages for t in tokenize(p["text"]))
    cards, seen = [], set()
    for n, p in enumerate(passages, start=1):
        for sent in studio.sentences(p["text"]):
            if len(cards) >= count:
                return cards
            if not 40 <= len(sent) <= 300:
                continue
            term = _key_term(sent, common)
            if not term or term.lower() in seen:
                continue
            seen.add(term.lower())
            cloze = re.sub(r"\b" + re.escape(term) + r"\b", "_____", sent, count=1)
            cards.append({"question": f"Fill in the blank: {cloze}", "answer": term, "hint": "",
                          "difficulty": difficulty if difficulty != "mixed" else "medium", "tags": ["cloze"],
                          "citations": [n]})
    return cards


def flashcards_prompt(topic: str, context: str, count: int, difficulty: str) -> str:
    level = {
        "easy": "definitions and basic recall",
        "medium": "understanding and how/why questions",
        "hard": "application, comparison and multi-step reasoning",
        "mixed": "a mix of easy recall, medium understanding and hard application",
    }[difficulty]
    return (
        f"Write {count} flashcards about '{topic}' using ONLY the numbered passages. Difficulty: {difficulty} "
        f"({level}). One idea per card, no duplicates, answers short (one to three sentences). Return ONLY JSON: "
        '{"cards": [{"question": ..., "answer": ..., "hint": short optional hint, "difficulty": "easy"|"medium"|'
        '"hard", "tags": [1-3 short topic tags], "citations": [passage numbers supporting the answer]}]}.\n\n'
        f"Passages:\n{context}"
    )


def generate_flashcards(kb, *, topic: str | None, sources=None, count: int = 10, difficulty: str = "mixed"):
    passages = studio.gather(kb, topic, sources, limit=60 if topic else 100)
    if not passages:
        return None
    subject = topic or "the material"
    context, reduced = map_reduce_context(subject, passages)
    cards = studio.llm_json(flashcards_prompt(subject, context, count, difficulty),
                            lambda d: validate_cards(d, len(passages), difficulty, count), CARD_SHAPE)
    fallback = not cards
    if fallback:
        cards = heuristic_cards(passages, count, difficulty)
    for i, c in enumerate(cards):
        c["id"] = f"c{i + 1}-" + hashlib.sha1(c["question"].encode()).hexdigest()[:8]
        c["sources"] = [x for x in (studio.citation(n, passages) for n in c["citations"]) if x]
    return {"title": f"Flashcards: {subject}", "topic": topic, "difficulty": difficulty, "cards": cards,
            "fallback": fallback, "map_reduce": reduced}
