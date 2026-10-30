"""Thin, thread-safe wrapper around the ``google-genai`` SDK.

Everything that talks to Gemini goes through this module, which gives us one place for
client construction, timeouts, retry/backoff on transient errors (429/5xx/timeouts),
and translation of SDK exceptions into :mod:`knowledge_forge.errors` types. Tests patch
the small public functions here instead of the SDK.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from typing import TypeVar

from knowledge_forge.config import get_settings
from knowledge_forge.errors import MISSING_KEY_MESSAGE, ConfigurationError, LLMError, RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_client = None
_client_key: str | None = None
_client_lock = threading.Lock()

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def get_client():
    """Return a cached ``genai.Client`` for the configured key (rebuilt if the key changes)."""
    global _client, _client_key
    settings = get_settings()
    key = settings.gemini_api_key
    if not key:
        raise ConfigurationError(MISSING_KEY_MESSAGE)
    with _client_lock:
        if _client is None or _client_key != key:
            from google import genai
            from google.genai import types

            _client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=settings.gemini_timeout_seconds * 1000),
            )
            _client_key = key
        return _client


def reset_client() -> None:
    global _client, _client_key
    with _client_lock:
        _client = None
        _client_key = None


def _status_of(exc: BaseException) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (ConfigurationError, RateLimitError)):
        return False
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    return any(word in name for word in ("timeout", "connect", "network", "remoteprotocol"))


def translate_error(exc: BaseException, what: str) -> Exception:
    """Map an SDK exception to a user-safe domain error (never echoes secrets)."""
    status = _status_of(exc)
    if status in (401, 403):
        return ConfigurationError("Gemini rejected the API key. Check GEMINI_API_KEY and its permissions.")
    if status == 429:
        return RateLimitError("Gemini rate limit reached. Wait a moment and try again.")
    if status == 404:
        return LLMError(f"The configured Gemini model for {what} was not found. Check GEN_MODEL/EMBED_MODEL.")
    return LLMError(f"Gemini {what} failed. Please try again.")


def with_retries(fn: Callable[[], T], what: str, attempts: int | None = None) -> T:
    """Run ``fn`` with exponential backoff + jitter on transient failures."""
    settings = get_settings()
    attempts = attempts or settings.llm_max_retries
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except (ConfigurationError, RateLimitError):
            raise
        except LLMError as exc:  # e.g. empty response: retry like a transient error
            last = exc
        except Exception as exc:
            last = exc
            if not is_retryable(exc):
                logger.warning("Gemini %s failed (not retryable): %s", what, type(exc).__name__)
                raise translate_error(exc, what) from exc
        logger.warning("Gemini %s attempt %d/%d failed: %s", what, attempt + 1, attempts, type(last).__name__)
        if attempt + 1 < attempts:
            delay = min(settings.llm_retry_max_seconds, settings.llm_retry_base_seconds * (2**attempt))
            time.sleep(delay * (0.5 + random.random() / 2))
    if isinstance(last, LLMError):
        raise last
    raise translate_error(last, what) if last else LLMError(f"Gemini {what} failed.")


def _gen_config(system: str | None, json_mode: bool):
    from google.genai import types

    settings = get_settings()
    kwargs: dict = {"temperature": settings.llm_temperature}
    if settings.llm_max_output_tokens:
        kwargs["max_output_tokens"] = settings.llm_max_output_tokens
    if system:
        kwargs["system_instruction"] = system
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    return types.GenerateContentConfig(**kwargs)


def generate(contents, system: str | None = None, json_mode: bool = False) -> str:
    """Generate text (non-streaming) with retries. Raises LLMError on empty output."""
    client = get_client()
    model = get_settings().gen_model

    def call() -> str:
        resp = client.models.generate_content(model=model, contents=contents, config=_gen_config(system, json_mode))
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            raise LLMError("The model returned an empty response.")
        return text

    return with_retries(call, "generation")


def generate_stream(contents, system: str | None = None) -> Iterator[str]:
    """Yield text deltas. Retries only happen before the first token is produced."""
    client = get_client()
    model = get_settings().gen_model

    def start():
        return client.models.generate_content_stream(model=model, contents=contents, config=_gen_config(system, False))

    stream = with_retries(start, "generation")
    try:
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
    except Exception as exc:
        raise translate_error(exc, "generation") from exc


def embed(texts: list[str], model: str, task_type: str | None, dim: int) -> list[list[float]]:
    """Embed ``texts`` with one API call (no retries here; callers decide)."""
    from google.genai import types

    config = types.EmbedContentConfig(output_dimensionality=dim, task_type=task_type)
    resp = get_client().models.embed_content(model=model, contents=texts, config=config)
    return [list(e.values or []) for e in (resp.embeddings or [])]


def upload_file(path: str, mime_type: str):
    return get_client().files.upload(file=path, config={"mime_type": mime_type})


def get_file(name: str):
    return get_client().files.get(name=name)


def delete_file(name: str) -> None:
    get_client().files.delete(name=name)
