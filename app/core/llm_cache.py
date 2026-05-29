import hashlib
import json
import logging
import re
import redis as redis_lib
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)
PLANNER_CACHE_TTL = 600  # 10 minutes (Requirement 5.3)
CACHE_KEY_PREFIX = "llm_cache:planner:"


def normalize_query(query: str) -> str:
    """
    Normalize query for cache key:
    - lowercase
    - strip leading/trailing whitespace
    - collapse internal whitespace
    - remove trailing punctuation

    Requirement 5.6
    """
    q = query.lower().strip()
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r'[?.!,;:]+$', '', q).strip()
    return q


def _cache_key(query: str) -> str:
    normalized = normalize_query(query)
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
    return f"{CACHE_KEY_PREFIX}{digest}"


class PlannerLLMCache:
    """
    Redis cache for PlannerAgent LLM decisions.
    Falls back to bypass (no cache) if Redis is unavailable.
    """

    def __init__(self) -> None:
        self._enabled = True
        self._hit_count = 0
        self._miss_count = 0
        try:
            self._redis = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
            self._redis.ping()
        except Exception as exc:
            logger.warning("PlannerLLMCache: Redis unavailable (%s), cache disabled", exc)
            self._enabled = False

    def get(self, query: str) -> Optional[dict]:
        if not self._enabled:
            return None
        try:
            raw = self._redis.get(_cache_key(query))
            if raw:
                self._hit_count += 1
                self._log_hit_rate()
                return json.loads(raw)
        except Exception as exc:
            logger.warning("PlannerLLMCache.get failed: %s", exc)
        self._miss_count += 1
        return None

    def set(self, query: str, decision: dict) -> None:
        if not self._enabled:
            return
        try:
            self._redis.set(_cache_key(query), json.dumps(decision), ex=PLANNER_CACHE_TTL)
        except Exception as exc:
            logger.warning("PlannerLLMCache.set failed: %s", exc)

    def _log_hit_rate(self) -> None:
        total = self._hit_count + self._miss_count
        if total > 0 and total % 100 == 0:  # Requirement 8.5
            rate = self._hit_count / total * 100
            logger.info("PlannerLLMCache hit rate: %.1f%% (%d/%d)", rate, self._hit_count, total)


_planner_cache = PlannerLLMCache()


def get_planner_cache() -> PlannerLLMCache:
    return _planner_cache
