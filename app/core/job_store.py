import json
import logging
from typing import Optional

from app.core.config import settings
from app.core.redis_client import create_redis_client, mask_redis_url

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 3600  # 1 hour
JOB_KEY_PREFIX = "job:"


class RedisJobStore:
    """
    Persistent async job store backed by Redis with in-memory fallback.

    Key structure: job:{task_id}  →  JSON blob
    TTL: 1 hour (refreshed on update)

    Fallback: if Redis is unavailable, uses an in-memory dict and logs a warning.
    This ensures the service degrades gracefully rather than failing hard.
    """

    def __init__(self) -> None:
        self._fallback: dict[str, dict] = {}
        self._use_redis = True
        try:
            self._redis = create_redis_client(decode_responses=True)
            self._redis.ping()
            logger.info(
                "RedisJobStore connected to Redis at %s",
                mask_redis_url(settings.REDIS_URL),
            )
        except Exception as exc:
            logger.warning(
                "RedisJobStore: Redis unavailable (%s), using in-memory fallback", exc
            )
            self._use_redis = False

    def create_job(self, task_id: str, initial_data: dict) -> None:
        """Create a new job entry with TTL=1h."""
        self._set(task_id, initial_data)

    def get_job(self, task_id: str) -> Optional[dict]:
        """Return job data or None if not found."""
        return self._get(task_id)

    def update_job(self, task_id: str, data: dict) -> None:
        """Update job data (refreshes TTL)."""
        self._set(task_id, data)

    def delete_job(self, task_id: str) -> None:
        """Delete job after result is delivered."""
        if self._use_redis:
            try:
                self._redis.delete(f"{JOB_KEY_PREFIX}{task_id}")
                return
            except Exception as exc:
                logger.warning("RedisJobStore.delete_job failed: %s", exc)
        self._fallback.pop(task_id, None)

    def _set(self, task_id: str, data: dict) -> None:
        if self._use_redis:
            try:
                self._redis.set(
                    f"{JOB_KEY_PREFIX}{task_id}",
                    json.dumps(data),
                    ex=JOB_TTL_SECONDS,
                )
                return
            except Exception as exc:
                logger.warning("RedisJobStore._set failed, using fallback: %s", exc)
        self._fallback[task_id] = data

    def _get(self, task_id: str) -> Optional[dict]:
        if self._use_redis:
            try:
                raw = self._redis.get(f"{JOB_KEY_PREFIX}{task_id}")
                if raw:
                    return json.loads(raw)
                return None
            except Exception as exc:
                logger.warning("RedisJobStore._get failed, using fallback: %s", exc)
        return self._fallback.get(task_id)


# Module-level singleton — shared by research.py and chatbot.py
_job_store = RedisJobStore()


def get_job_store() -> RedisJobStore:
    return _job_store
