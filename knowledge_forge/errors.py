"""Domain-specific exceptions mapped to HTTP responses by the web layer."""


class KnowledgeForgeError(Exception):
    """Base error; ``status`` is the suggested HTTP status code."""

    status = 500


class ConfigurationError(KnowledgeForgeError):
    """Required configuration (e.g. GEMINI_API_KEY) is missing."""

    status = 503


class ValidationError(KnowledgeForgeError):
    """User input is invalid."""

    status = 400


class IngestionError(KnowledgeForgeError):
    """Content could not be extracted from a source."""

    status = 422


class LLMError(KnowledgeForgeError):
    """The language model failed or returned unusable output."""

    status = 502


class RateLimitError(KnowledgeForgeError):
    """Too many requests (from this client, or upstream at Gemini)."""

    status = 429


class NotFoundError(KnowledgeForgeError):
    """A referenced resource (e.g. a source) does not exist."""

    status = 404


MISSING_KEY_MESSAGE = (
    "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey and "
    "export it (or add it to a local .env file), then restart the server."
)
