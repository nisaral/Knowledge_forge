"""Structured logging with secret redaction."""
from __future__ import annotations

import json
import logging
import re

from knowledge_forge.config import Settings, get_settings

_KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"(?i)\b(api[_-]?key|key|access[_-]?token|token|secret|password)=([^&\s]+)"),
]


class RedactingFilter(logging.Filter):
    """Replace configured secret values and anything that looks like an API key."""

    def __init__(self, secrets: list[str] | None = None) -> None:
        super().__init__()
        self.secrets = [s for s in (secrets or []) if s and len(s) >= 8]

    def redact(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        text = _KEY_PATTERNS[0].sub("[REDACTED]", text)
        return _KEY_PATTERNS[1].sub(lambda m: f"{m.group(1)}=[REDACTED]", text)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        record.msg = self.redact(message)
        record.args = None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id", "path", "status", "duration_ms"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_text:
            data["exc"] = record.exc_text
        return json.dumps(data, ensure_ascii=False)


_configured = False


def configure_logging(settings: Settings | None = None, force: bool = False) -> None:
    """Install a redacting handler on the root logger (idempotent)."""
    global _configured
    if _configured and not force:
        return
    settings = settings or get_settings()
    root = logging.getLogger()
    handler = logging.StreamHandler()
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter([settings.gemini_api_key, settings.qdrant_api_key]))
    for old in [h for h in root.handlers if getattr(h, "_kf", False)]:
        root.removeHandler(old)
    handler._kf = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    _configured = True
