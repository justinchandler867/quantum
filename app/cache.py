"""
Cache Layer
Redis-backed caching for correlation matrices and volatility vectors.
Falls back to in-memory dict if Redis is unavailable.
"""
import json
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# In-memory fallback cache
_mem_cache: dict[str, tuple[str, float]] = {}

_redis = None


def _get_redis():
    """Lazy Redis connection. Returns None if unavailable."""
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis as redis_lib
        from app.config import REDIS_URL
        _redis = redis_lib.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
        logger.info("Redis connected")
        return _redis
    except Exception as e:
        logger.warning(f"Redis unavailable ({e}), using in-memory cache")
        _redis = False  # sentinel: don't retry
        return None


def _universe_key(tickers: list[str]) -> str:
    """Hash key for a set of tickers (order-independent)."""
    return "|".join(sorted(tickers))


def cache_key(prefix: str, tickers: list[str], **kwargs) -> str:
    """Build a namespaced cache key."""
    parts = [prefix, _universe_key(tickers)]
    for k, v in sorted(kwargs.items()):
        parts.append(f"{k}={v}")
    return ":".join(parts)


def cache_set(key: str, value: dict, ttl: int = 3600) -> None:
    """Store a JSON-serializable dict."""
    serialized = json.dumps(value, default=_json_serialize)
    r = _get_redis()
    if r:
        try:
            r.setex(key, ttl, serialized)
            return
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
    # Fallback to memory
    import time
    _mem_cache[key] = (serialized, time.time() + ttl)


def cache_get(key: str) -> Optional[dict]:
    """Retrieve a cached dict. Returns None on miss."""
    r = _get_redis()
    if r:
        try:
            raw = r.get(key)
            if raw:
                return json.loads(raw)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")

    # Fallback to memory
    import time
    entry = _mem_cache.get(key)
    if entry:
        raw, expires = entry
        if time.time() < expires:
            return json.loads(raw)
        else:
            del _mem_cache[key]
    return None


def cache_invalidate(prefix: str) -> None:
    """Invalidate all keys matching a prefix."""
    r = _get_redis()
    if r:
        try:
            keys = r.keys(f"{prefix}:*")
            if keys:
                r.delete(*keys)
            return
        except Exception as e:
            logger.warning(f"Redis invalidate failed: {e}")
    # Fallback: clear matching memory keys
    to_del = [k for k in _mem_cache if k.startswith(prefix)]
    for k in to_del:
        del _mem_cache[k]


def _json_serialize(obj):
    """Custom serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Not serializable: {type(obj)}")
