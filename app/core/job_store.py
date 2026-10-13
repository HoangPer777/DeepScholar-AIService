import json
import logging
import threading
from typing import Optional

from redis.exceptions import WatchError

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
        self._fallback_lock = threading.Lock()
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

    def create_job_if_absent(self, task_id: str, initial_data: dict) -> bool:
        """Atomically claim a task key without changing legacy job semantics."""
        key = f"{JOB_KEY_PREFIX}{task_id}"
        if self._use_redis:
            try:
                return bool(
                    self._redis.set(
                        key,
                        json.dumps(initial_data),
                        ex=JOB_TTL_SECONDS,
                        nx=True,
                    )
                )
            except Exception as exc:
                logger.warning("RedisJobStore.create_job_if_absent failed: %s", exc)
                raise

        with self._fallback_lock:
            if task_id in self._fallback:
                return False
            self._fallback[task_id] = initial_data
            return True

    def get_job(self, task_id: str) -> Optional[dict]:
        """Return job data or None if not found."""
        return self._get(task_id)

    def update_job(self, task_id: str, data: dict) -> None:
        """Update job data (refreshes TTL)."""
        self._set(task_id, data)

    def get_pdf_job(self, task_id: str) -> Optional[dict]:
        """Read a PDF job only from shared Redis; never use process-local fallback."""
        if not self._use_redis:
            raise RuntimeError("Persistent PDF job storage is unavailable")
        try:
            raw = self._redis.get(f"{JOB_KEY_PREFIX}{task_id}")
        except Exception as exc:
            logger.warning("RedisJobStore.get_pdf_job failed")
            raise RuntimeError("Persistent PDF job storage is unavailable") from exc
        if not raw:
            return None
        try:
            job = json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.warning("RedisJobStore found malformed PDF job state")
            raise RuntimeError("Persistent PDF job storage is unavailable") from exc
        return job if isinstance(job, dict) and job.get("kind") == "pdf" else None

    def update_pdf_job(self, task_id: str, updates: dict) -> Optional[dict]:
        """Atomically merge a PDF snapshot update, preserving identity and terminal state."""
        if not self._use_redis:
            raise RuntimeError("Persistent PDF job storage is unavailable")
        key = f"{JOB_KEY_PREFIX}{task_id}"
        try:
            for _ in range(5):
                with self._redis.pipeline() as pipe:
                    try:
                        pipe.watch(key)
                        raw = pipe.get(key)
                        if not raw:
                            return None
                        current = json.loads(raw)
                        if not isinstance(current, dict) or current.get("kind") != "pdf":
                            return None
                        if current.get("status") in {"completed", "failed"}:
                            return current
                        merged = {**current, **updates}
                        merged["kind"] = "pdf"
                        for identity in ("task_id", "article_id", "generation", "content_revision", "idempotency_key", "slug"):
                            if identity in current:
                                merged[identity] = current[identity]
                        pipe.multi()
                        pipe.set(key, json.dumps(merged), ex=JOB_TTL_SECONDS)
                        pipe.execute()
                        return merged
                    except WatchError:
                        continue
            raise RuntimeError("PDF job changed too frequently to update safely")
        except Exception as exc:
            logger.warning("RedisJobStore.update_pdf_job failed")
            raise RuntimeError("Persistent PDF job storage is unavailable") from exc

    def delete_job(self, task_id: str) -> None:
        """Delete job after result is delivered."""
        if self._use_redis:
            try:
                self._redis.delete(f"{JOB_KEY_PREFIX}{task_id}")
                return
            except Exception as exc:
                logger.warning("RedisJobStore.delete_job failed: %s", exc)
        self._fallback.pop(task_id, None)

    @property
    def is_persistent(self) -> bool:
        """Whether job state is currently backed by Redis."""
        return self._use_redis

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
