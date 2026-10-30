"""Redis-backed cache coordination for multi-instance BM25 sync."""
from __future__ import annotations

import logging
import threading
import time

from knowledge_forge.config import get_settings

logger = logging.getLogger(__name__)

_redis_client = None
_redis_failed_at = 0.0
_RETRY_SECONDS = 30.0
_local_version = 0
_lock = threading.Lock()


def _get_redis():
    """Return a connected Redis client, or None. Thread-safe; retries every 30 s after a failure."""
    global _redis_client, _redis_failed_at
    if _redis_client is not None:
        return _redis_client
    if not get_settings().redis_url:
        return None
    with _lock:
        if _redis_client is not None:
            return _redis_client
        if _redis_failed_at and time.monotonic() - _redis_failed_at < _RETRY_SECONDS:
            return None
        try:
            import redis

            client = redis.from_url(get_settings().redis_url, decode_responses=True, socket_timeout=5)
            client.ping()
            _redis_client = client
            logger.info("Connected to Redis for KB cache coordination.")
            return client
        except Exception as exc:
            logger.warning("Redis unavailable (%s); using in-process cache only.", type(exc).__name__)
            _redis_failed_at = time.monotonic()
            return None


def get_kb_version() -> int:
    client = _get_redis()
    if client:
        try:
            val = client.get(get_settings().kb_cache_version_key)
            return int(val) if val else 0
        except Exception as exc:
            logger.warning("Redis get version failed: %s", type(exc).__name__)
    return _local_version


def bump_kb_version() -> int:
    client = _get_redis()
    if client:
        try:
            return int(client.incr(get_settings().kb_cache_version_key))
        except Exception as exc:
            logger.warning("Redis bump version failed: %s", type(exc).__name__)
    global _local_version
    with _lock:
        _local_version += 1
        return _local_version


def reset_kb_version() -> None:
    client = _get_redis()
    if client:
        try:
            client.set(get_settings().kb_cache_version_key, 0)
            return
        except Exception as exc:
            logger.warning("Redis reset version failed: %s", type(exc).__name__)
    global _local_version
    _local_version = 0


def redis_health_check() -> bool:
    client = _get_redis()
    if not client:
        return not get_settings().redis_url
    try:
        client.ping()
        return True
    except Exception:
        return False


def reset_cache_client() -> None:
    """Drop the cached Redis client (used by tests and after config changes)."""
    global _redis_client, _local_version, _redis_failed_at
    with _lock:
        _redis_client = None
        _redis_failed_at = 0.0
        _local_version = 0
