import hashlib
import json
from typing import Any, Optional

from redis import Redis

from core.config import get_settings

_client: Optional[Redis] = None


def _redis() -> Redis:
    global _client
    if _client is None:
        settings = get_settings()
        _client = Redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def make_cache_key(prefix: str, payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"dr:{prefix}:{digest}"


def cache_get(key: str) -> Optional[dict]:
    settings = get_settings()
    if not settings.use_cache:
        return None
    val = _redis().get(key)
    if not val:
        return None
    return json.loads(val)


def cache_set(key: str, value: dict, ttl_seconds: Optional[int] = None) -> None:
    settings = get_settings()
    if not settings.use_cache:
        return
    ttl = ttl_seconds or settings.cache_ttl_seconds
    _redis().setex(key, ttl, json.dumps(value, ensure_ascii=False))


def cache_delete_prefix(prefix: str) -> int:
    r = _redis()
    count = 0
    for key in r.scan_iter(match=f"dr:{prefix}:*"):
        r.delete(key)
        count += 1
    return count
