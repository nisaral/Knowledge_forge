"""Prompt builders. Pure functions so they can be unit tested without the network."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

TOOLS = ("mock_test", "mindmap", "flowchart", "storyboard", "flashcards", "study_plan")

_MERMAID_RULES = (
    "MERMAID RULES: 1) Use simple single-word IDs (A, B, C1). "
    '2) Wrap node text with special chars in double quotes: A["Text (with parens)"]. '
    "3) Do not use HTML or click handlers."
)


def format_context(hits: Sequence[dict]) -> str:
    """Render retrieved chunks as numbered passages so the model can cite ``[n]``."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        source = hit.get("source", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{hit.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


def answer_prompt(question: str, hits: Sequence[dict]) -> str:
    return (
        "You are a helpful educational assistant. Answer using ONLY the numbered context "
        "passages below. Cite passages inline like [1] or [2][3] after the sentences they "
        "support. If the context does not contain the answer, say so honestly.\n\n"
        f"Context:\n{format_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer in clear, well-structured markdown."
    )


def chat_prompt(message: str, hits: Sequence[dict], history: Iterable[dict]) -> str:
    history_text = "\n".join(
        f"{'User' if h.get('role') == 'user' else 'Assistant'}: {h.get('content', '')}"
        for h in history
    ) or "(no previous messages)"
    context = format_context(hits) if hits else "No relevant passages were retrieved."
    return (
        "You are a knowledgeable educational assistant. Ground your reply in the numbered "
        "context passages when they are relevant and cite them inline like [1]. If the "
        "context is not relevant, say that the knowledge base does not cover it before "
        "answering from general knowledge.\n\n"
        f"Knowledge Context:\n{context}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"User: {message}\n\n"
        "Respond clearly and helpfully in markdown."
    )


def summary_prompt(text: str) -> str:
    return (
        "Create a comprehensive, well-structured summary of the following educational content. "
        "Use headings, bullet points, and a short 'Key takeaways' section. Format in markdown.\n\n"
        f"{text}"
    )


def rerank_prompt(query: str, passages: Sequence[str], k: int) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(passages))
    return (
        f"Given the query: '{query}'\n\n"
        f"Here are {len(passages)} candidate passages:\n{numbered}\n\n"
        f"Return ONLY a JSON array of the {k} passage numbers (1-indexed) most relevant to "
        "the query, ordered by relevance. Example: [2,5,1]"
    )


def entities_prompt(text: str) -> str:
    return (
        "Extract named entities from the text below. Return a JSON object with keys "
        "'Person', 'Organization', 'Location', 'Concept', each a list of unique names. "
        "Use ```json ... ```.\n\n"
        f"Text:\n{text}"
    )


def cluster_label_prompt(sample: str) -> str:
    return f"Give a concise 2-3 word topic label (no quotes, no punctuation) for: {sample[:500]}"


def tool_prompt(tool: str, topic: str, context: str) -> str:
    if tool not in TOOLS:
        raise ValueError(f"Unknown tool: {tool}")
    prompts = {
        "mock_test": (
            f"Based on the context, generate a 5-question multiple-choice test on '{topic}'. "
            "Provide 4 options (A-D) per question. End with an 'Answers:' section that "
            f"briefly explains each answer.\n\nContext:\n{context}"
        ),
        "mindmap": (
            f"Create a Mermaid.js 'mindmap' about '{topic}' from the context.\n"
            f"{_MERMAID_RULES}\nContext:\n{context}\nOutput ONLY ```mermaid ... ```"
        ),
        "flowchart": (
            f"Create a Mermaid.js 'graph TD' flowchart for '{topic}' from the context.\n"
            f"{_MERMAID_RULES}\nContext:\n{context}\nOutput ONLY ```mermaid ... ```"
        ),
        "storyboard": (
            f"Create a 3-panel storyboard about '{topic}'. For each panel give 'title' "
            "(a few words) and 'scene' (2-3 sentence description). Return ONLY a valid JSON "
            f"array inside ```json ... ```.\n\nContext:\n{context}"
        ),
        "flashcards": (
            f"Generate 8 flashcards about '{topic}' from the context. Each flashcard has a "
            '"question" and an "answer". Return ONLY a JSON array: '
            '[{"question":"...","answer":"..."}] inside ```json ... ```.'
            f"\n\nContext:\n{context}"
        ),
        "study_plan": (
            f"Create a structured 4-week study plan to master '{topic}' based on the context. "
            "Format as markdown with week headings, daily goals, and resources. Be specific "
            f"and actionable.\n\nContext:\n{context}"
        ),
    }
    return prompts[tool]
