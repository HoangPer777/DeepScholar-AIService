"""
Unit tests for Deep Research Optimization components.

Tests cover:
- RedisJobStore.delete_job removes the entry (Requirement 2.3)
- RedisJobStore falls back to in-memory dict when Redis raises ConnectionError (Requirement 2.4)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import redis as redis_lib


# ---------------------------------------------------------------------------
# RedisJobStore — delete_job and in-memory fallback
# ---------------------------------------------------------------------------


class TestRedisJobStoreDeleteJob:
    """Tests for Requirement 2.3: delete_job removes the stored entry."""

    def test_job_store_deletes_after_fetch(self):
        """
        Create a job, retrieve it, delete it, then verify get_job returns None.

        Requirement 2.3: delete_job removes the entry from the store.
        """
        # Patch redis.from_url so RedisJobStore uses a MagicMock Redis client
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        # Simulate Redis get/set/delete behaviour with an in-process dict
        _store: dict[str, str] = {}

        def fake_set(key, value, ex=None):
            _store[key] = value

        def fake_get(key):
            return _store.get(key)

        def fake_delete(key):
            _store.pop(key, None)

        mock_redis.set.side_effect = fake_set
        mock_redis.get.side_effect = fake_get
        mock_redis.delete.side_effect = fake_delete

        with patch("redis.from_url", return_value=mock_redis):
            from app.core.job_store import RedisJobStore

            store = RedisJobStore()

        task_id = "test-task-delete-001"
        job_data = {"status": "done", "result": {"answer": "hello"}}

        # Create and verify the job exists
        store.create_job(task_id, job_data)
        retrieved = store.get_job(task_id)
        assert retrieved is not None, "Job should exist after create_job"
        assert retrieved["status"] == "done"

        # Delete and verify it is gone
        store.delete_job(task_id)
        after_delete = store.get_job(task_id)
        assert after_delete is None, "get_job should return None after delete_job"


class TestRedisJobStoreInMemoryFallback:
    """Tests for Requirement 2.4: in-memory fallback activates on ConnectionError."""

    def test_job_store_redis_fallback(self):
        """
        When Redis raises ConnectionError on ping(), RedisJobStore must switch to
        the in-memory fallback dict so that create_job / get_job still work.

        Requirement 2.4: in-memory fallback activates when Redis is unavailable.
        """
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = redis_lib.ConnectionError("Redis not available")

        with patch("redis.from_url", return_value=mock_redis):
            from app.core.job_store import RedisJobStore

            store = RedisJobStore()

        # _use_redis must be False after a ConnectionError on ping
        assert store._use_redis is False, (
            "RedisJobStore should set _use_redis=False when ping raises ConnectionError"
        )

        task_id = "test-task-fallback-001"
        job_data = {"status": "pending", "session_id": "sess-abc"}

        # create_job and get_job must still work via the in-memory fallback
        store.create_job(task_id, job_data)
        result = store.get_job(task_id)

        assert result is not None, "get_job should return data via in-memory fallback"
        assert result["status"] == "pending"
        assert result["session_id"] == "sess-abc"

    def test_job_store_fallback_delete_works(self):
        """
        When using the in-memory fallback, delete_job should also remove the entry.

        Requirement 2.4: fallback dict behaves consistently with Redis-backed store.
        """
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = redis_lib.ConnectionError("Redis not available")

        with patch("redis.from_url", return_value=mock_redis):
            from app.core.job_store import RedisJobStore

            store = RedisJobStore()

        task_id = "test-task-fallback-delete-001"
        store.create_job(task_id, {"status": "done"})

        # Confirm it exists
        assert store.get_job(task_id) is not None

        # Delete and confirm it is gone
        store.delete_job(task_id)
        assert store.get_job(task_id) is None, (
            "delete_job should remove entry from in-memory fallback"
        )


# ---------------------------------------------------------------------------
# PlannerLLMCache — cache hit and query normalization
# ---------------------------------------------------------------------------


class TestPlannerLLMCacheCacheHit:
    """Tests for Requirement 5.2: cache hit returns the stored decision."""

    def test_planner_cache_hit(self):
        """
        After set("machine learning", decision), get("machine learning") must
        return the same decision dict.

        Requirement 5.2: cache hit returns stored decision.
        """
        # Use an in-process dict to simulate Redis get/set behaviour
        _store: dict[str, str] = {}

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        def fake_set(key, value, ex=None):
            _store[key] = value

        def fake_get(key):
            return _store.get(key)

        mock_redis.set.side_effect = fake_set
        mock_redis.get.side_effect = fake_get

        with patch("redis.from_url", return_value=mock_redis):
            from app.core.llm_cache import PlannerLLMCache

            cache = PlannerLLMCache()

        decision = {
            "need_clarification": False,
            "need_external_search": True,
            "focus_sections": ["methodology", "results"],
            "search_queries": ["machine learning overview"],
        }

        cache.set("machine learning", decision)
        result = cache.get("machine learning")

        assert result is not None, "Cache should return a value on hit"
        assert result == decision, "Returned decision must equal the stored decision"


class TestPlannerLLMCacheNormalization:
    """Tests for Requirement 5.6: queries differing only in case/whitespace share the same cache entry."""

    def test_planner_cache_normalization(self):
        """
        set("  Machine Learning  ", decision) then get("machine learning") must
        return the same decision because both queries normalize to the same key.

        Requirement 5.6: queries differing only in case/whitespace hit the same cache entry.
        """
        _store: dict[str, str] = {}

        mock_redis = MagicMock()
        mock_redis.ping.return_value = True

        def fake_set(key, value, ex=None):
            _store[key] = value

        def fake_get(key):
            return _store.get(key)

        mock_redis.set.side_effect = fake_set
        mock_redis.get.side_effect = fake_get

        with patch("redis.from_url", return_value=mock_redis):
            from app.core.llm_cache import PlannerLLMCache

            cache = PlannerLLMCache()

        decision = {
            "need_clarification": False,
            "need_external_search": True,
            "focus_sections": ["introduction"],
            "search_queries": ["machine learning basics"],
        }

        # Store with padded, mixed-case query
        cache.set("  Machine Learning  ", decision)

        # Retrieve with lowercase, no-padding query — must hit the same cache entry
        result = cache.get("machine learning")

        assert result is not None, (
            "Cache should return a value: '  Machine Learning  ' and 'machine learning' "
            "must normalize to the same cache key"
        )
        assert result == decision, "Returned decision must equal the stored decision"


# ---------------------------------------------------------------------------
# Chat API / state / frontend contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_chat_404_no_context():
    """
    Follow-up chat jobs with a provided session_id must return a 404-style job
    error when that session has no Research_Context.

    Requirement 3.5
    """
    from app.api import chatbot

    with (
        patch("app.api.chatbot._has_research_context", return_value=False),
        patch.object(chatbot._job_store, "update_job") as update_job,
    ):
        await chatbot._run_chat_job(
            task_id="task-no-context",
            question="What does the report imply?",
            article_id=None,
            session_id="missing-session",
            require_context=True,
        )

    update_job.assert_called_once()
    task_id, payload = update_job.call_args.args
    assert task_id == "task-no-context"
    assert payload["status"] == "error"
    assert payload["error_code"] == 404
    assert "research context" in payload["error"].lower()


def test_agent_state_default_max_iterations():
    """
    AgentState must default max_iterations to 1 to avoid unnecessary
    Writer-Reviewer loops.

    Requirement 1.6
    """
    from app.workflows.states import AgentState

    state = AgentState(question="test question")
    assert state.max_iterations == 1


def test_follow_up_response_has_is_fast_chat_field():
    """
    Frontend FollowUpResponse must include the optional is_fast_chat field so
    the UI can render the Fast Reply badge.

    Requirement 6.3
    """
    repo_root = Path(__file__).resolve().parents[2]
    service_file = repo_root / "DeepScholar-Frontend" / "services" / "research.ts"
    content = service_file.read_text(encoding="utf-8")

    assert "export interface FollowUpResponse" in content
    assert "is_fast_chat?: boolean" in content
