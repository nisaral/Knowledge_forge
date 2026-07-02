"""Redis-backed cache coordination for multi-instance BM25 sync."""
import logging

from config import KB_CACHE_VERSION_KEY, REDIS_URL

logger = logging.getLogger(__name__)

_redis_client = None
_local_version = 0


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not REDIS_URL:
        return None
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        logger.info("Connected to Redis for KB cache coordination.")
        return _redis_client
    except Exception as exc:
        logger.warning("Redis unavailable (%s); using in-process cache only.", exc)
        _redis_client = False
        return None


def get_kb_version() -> int:
    client = _get_redis()
    if client:
        try:
            val = client.get(KB_CACHE_VERSION_KEY)
            return int(val) if val else 0
        except Exception as exc:
            logger.warning("Redis get version failed: %s", exc)
    return _local_version


def bump_kb_version() -> int:
    client = _get_redis()
    if client:
        try:
            return int(client.incr(KB_CACHE_VERSION_KEY))
        except Exception as exc:
            logger.warning("Redis bump version failed: %s", exc)
    global _local_version
    _local_version += 1
    return _local_version


def reset_kb_version() -> None:
    client = _get_redis()
    if client:
        try:
            client.set(KB_CACHE_VERSION_KEY, 0)
            return
        except Exception as exc:
            logger.warning("Redis reset version failed: %s", exc)
    global _local_version
    _local_version = 0


def redis_health_check() -> bool:
    client = _get_redis()
    if not client:
        return not REDIS_URL
    try:
        client.ping()
        return True
    except Exception:
        return False