import pytest

from knowledge_forge import llm
from knowledge_forge.errors import ConfigurationError, LLMError
from knowledge_forge.prompts import TOOLS, answer_prompt, chat_prompt, format_context, tool_prompt


def test_clean_llm_output_strips_fences():
    assert llm.clean_llm_output("```mermaid\ngraph TD\nA-->B\n```", "mermaid").startswith("graph TD")
    assert llm.clean_llm_output('```json\n[1]\n```', "json") == "[1]"


def test_parse_json_output_tolerates_noise():
    assert llm.parse_json_output('Sure!\n```json\n[{"a": 1},]\n```') == [{"a": 1}]
    with pytest.raises(LLMError):
        llm.parse_json_output("definitely not json")


@pytest.mark.parametrize(
    "raw, expected",
    [("3, 1, 2", [2, 0, 1]), ("[2,2,9,1]", [1, 0]), ("none", []), ("0, 1", [0])],
)
def test_parse_rerank(raw, expected):
    assert llm.parse_rerank(raw, n_candidates=4, k=3) == expected


def test_sanitize_mermaid_quotes_labels():
    out = llm.sanitize_mermaid("graph TD\nA[Start (here)] --> B{Ok?}")
    assert out.startswith("graph TD")
    assert '"' in out


def test_ensure_configured_requires_key(monkeypatch):
    from knowledge_forge import config

    monkeypatch.delenv("GEMINI_API_KEY")
    config.get_settings.cache_clear()
    with pytest.raises(ConfigurationError):
        llm.ensure_gemini_configured()


def test_generate_content_retries_then_fails(monkeypatch):
    calls = []

    class Model:
        def __init__(self, *_a, **_k):
            pass

        def generate_content(self, _prompt):
            calls.append(1)
            raise RuntimeError("boom")

    monkeypatch.setattr(llm.genai, "GenerativeModel", Model)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    with pytest.raises(LLMError):
        llm.generate_content("hi")
    assert len(calls) == 3


def test_prompts_include_numbered_context():
    hits = [{"text": "alpha", "source": "doc-a"}, {"text": "beta", "source": "doc-b"}]
    ctx = format_context(hits)
    assert "[1]" in ctx and "[2]" in ctx and "doc-b" in ctx
    assert "alpha" in answer_prompt("q?", hits)
    prompt = chat_prompt("next?", hits, [{"role": "user", "content": "earlier question"}])
    assert "earlier question" in prompt


def test_tool_prompt_known_and_unknown():
    for tool in TOOLS:
        assert "topic-x" in tool_prompt(tool, "topic-x", "ctx")
    with pytest.raises(ValueError):
        tool_prompt("nope", "t", "c")
