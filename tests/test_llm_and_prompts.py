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


class _FakeModels:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        return self.behaviour(len(self.calls))


class _FakeClient:
    def __init__(self, behaviour):
        self.models = _FakeModels(behaviour)


class _ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"api error {code}")
        self.code = code


def _install(monkeypatch, behaviour):
    from knowledge_forge import gemini

    client = _FakeClient(behaviour)
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    monkeypatch.setattr(gemini.time, "sleep", lambda _s: None)
    return client


def test_generate_content_retries_then_fails(monkeypatch):
    def always_500(_n):
        raise _ApiError(503)

    client = _install(monkeypatch, always_500)
    with pytest.raises(LLMError):
        llm.generate_content("hi")
    assert len(client.models.calls) == 3


def test_generate_content_recovers_after_transient_error(monkeypatch):
    class Resp:
        text = " hello "

    def flaky(n):
        if n == 1:
            raise _ApiError(429)
        return Resp()

    client = _install(monkeypatch, flaky)
    assert llm.generate_content("hi", system="sys", json_mode=True) == "hello"
    config = client.models.calls[-1][2]
    assert config.system_instruction == "sys" and config.response_mime_type == "application/json"


def test_generate_content_does_not_retry_auth_errors(monkeypatch):
    from knowledge_forge.errors import ConfigurationError as CE

    def denied(_n):
        raise _ApiError(403)

    client = _install(monkeypatch, denied)
    with pytest.raises(CE) as info:
        llm.generate_content("hi")
    assert len(client.models.calls) == 1
    assert "test-key" not in str(info.value)


def test_retry_backoff_is_capped(monkeypatch):
    from knowledge_forge import config, gemini

    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_RETRY_BASE_SECONDS", "2")
    monkeypatch.setenv("LLM_RETRY_MAX_SECONDS", "3")
    config.get_settings.cache_clear()
    sleeps = []
    monkeypatch.setattr(gemini.time, "sleep", sleeps.append)

    def fail():
        raise TimeoutError("slow")

    with pytest.raises(LLMError):
        gemini.with_retries(fail, "generation")
    assert len(sleeps) == 4 and max(sleeps) <= 3


def test_empty_response_is_an_error(monkeypatch):
    class Resp:
        text = ""

    _install(monkeypatch, lambda _n: Resp())
    with pytest.raises(LLMError):
        llm.generate_content("hi")


def test_prompts_include_numbered_context():
    hits = [{"text": "alpha", "source": "doc-a"}, {"text": "beta", "source": "doc-b"}]
    ctx = format_context(hits)
    assert 'n="1"' in ctx and 'n="2"' in ctx and "doc-b" in ctx
    assert "alpha" in answer_prompt("q?", hits)
    prompt = chat_prompt("next?", hits, [{"role": "user", "content": "earlier question"}])
    assert "earlier question" in prompt


def test_tool_prompt_known_and_unknown():
    for tool in TOOLS:
        assert "topic-x" in tool_prompt(tool, "topic-x", "ctx")
    with pytest.raises(ValueError):
        tool_prompt("nope", "t", "c")
