"""
Unit tests for MemoryStore Redis operations.

Tests use ``fakeredis`` to run without a live Redis instance.

Requirements: 2.1, 2.2, 2.3, 2.6, 12.5
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import fakeredis
import pytest

from app.core.memory_store import (
    DEFAULT_MAX_MESSAGES,
    SESSION_TTL_SECONDS,
    MemoryStore,
    SessionContextNotFoundError,
    _context_key,
    _user_sessions_key,
)
from app.schemas.chat_models import (
    ContextWindow,
    Message,
    MessageRole,
    ResearchReport,
    Source,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis():
    """Return a fakeredis server instance shared across a test."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server)
    yield client
    client.flushall()
    client.close()


@pytest.fixture
def store(fake_redis):
    """Return a MemoryStore backed by fakeredis."""
    return MemoryStore(redis_client=fake_redis)


def _make_message(
    session_id: str,
    role: MessageRole = MessageRole.USER,
    content: str = "Hello",
    token_count: int = 5,
) -> Message:
    return Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role=role,
        content=content,
        timestamp=datetime.utcnow(),
        token_count=token_count,
    )


def _make_report(answer: str = "Research answer") -> ResearchReport:
    return ResearchReport(
        answer=answer,
        sources=[],
        planner_decision={},
        confidence_score=0.9,
        review_feedback="Good",
    )


def _make_source(index: int = 1) -> Source:
    return Source(
        index=index,
        type="url",
        title=f"Source {index}",
        url=f"https://example.com/{index}",
    )


# ---------------------------------------------------------------------------
# save_message tests
# ---------------------------------------------------------------------------


class TestSaveMessage:
    def test_message_is_stored_in_redis(self, store, fake_redis):
        """Saving a message should create a Hash entry in Redis."""
        sid = str(uuid.uuid4())
        msg = _make_message(sid)
        store.save_message(sid, msg)

        raw = fake_redis.hget(_context_key(sid), "messages")
        assert raw is not None
        messages = json.loads(raw)
        assert len(messages) == 1
        assert messages[0]["message_id"] == msg.message_id

    def test_multiple_messages_are_appended(self, store, fake_redis):
        """Each call to save_message should append, not overwrite."""
        sid = str(uuid.uuid4())
        for i in range(3):
            store.save_message(sid, _make_message(sid, content=f"msg {i}"))

        raw = fake_redis.hget(_context_key(sid), "messages")
        messages = json.loads(raw)
        assert len(messages) == 3

    def test_total_tokens_is_updated(self, store, fake_redis):
        """total_tokens field should reflect the sum of all message token counts."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid, token_count=10))
        store.save_message(sid, _make_message(sid, token_count=20))

        raw = fake_redis.hget(_context_key(sid), "total_tokens")
        assert int(raw) == 30

    def test_ttl_is_set_to_24_hours(self, store, fake_redis):
        """The context key TTL should be set to SESSION_TTL_SECONDS."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))

        ttl = fake_redis.ttl(_context_key(sid))
        # Allow a small delta for test execution time
        assert SESSION_TTL_SECONDS - 5 <= ttl <= SESSION_TTL_SECONDS

    def test_user_session_registered_when_user_id_provided(self, store, fake_redis):
        """When user_id is given, session should appear in user's sorted set."""
        sid = str(uuid.uuid4())
        uid = "user-abc"
        store.save_message(sid, _make_message(sid), user_id=uid)

        members = fake_redis.zrange(_user_sessions_key(uid), 0, -1)
        decoded = [m.decode() if isinstance(m, bytes) else m for m in members]
        assert sid in decoded

    def test_no_user_session_registered_without_user_id(self, store, fake_redis):
        """When user_id is omitted, no user sorted set should be created."""
        sid = str(uuid.uuid4())
        uid = "user-xyz"
        store.save_message(sid, _make_message(sid))  # no user_id

        exists = fake_redis.exists(_user_sessions_key(uid))
        assert exists == 0

    def test_message_role_is_preserved(self, store, fake_redis):
        """The role field of the saved message should be preserved."""
        sid = str(uuid.uuid4())
        msg = _make_message(sid, role=MessageRole.ASSISTANT, content="Answer")
        store.save_message(sid, msg)

        raw = fake_redis.hget(_context_key(sid), "messages")
        messages = json.loads(raw)
        assert messages[0]["role"] == MessageRole.ASSISTANT.value

    def test_message_content_is_preserved(self, store, fake_redis):
        """The content field of the saved message should be preserved."""
        sid = str(uuid.uuid4())
        content = "What is quantum entanglement?"
        msg = _make_message(sid, content=content)
        store.save_message(sid, msg)

        raw = fake_redis.hget(_context_key(sid), "messages")
        messages = json.loads(raw)
        assert messages[0]["content"] == content


# ---------------------------------------------------------------------------
# get_context_window tests
# ---------------------------------------------------------------------------


class TestGetContextWindow:
    def test_raises_when_session_not_found(self, store):
        """Should raise SessionContextNotFoundError for unknown session."""
        with pytest.raises(SessionContextNotFoundError):
            store.get_context_window(str(uuid.uuid4()))

    def test_returns_context_window_model(self, store):
        """Should return a ContextWindow Pydantic model."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))
        result = store.get_context_window(sid)
        assert isinstance(result, ContextWindow)

    def test_returns_correct_session_id(self, store):
        """ContextWindow.session_id should match the requested session."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))
        result = store.get_context_window(sid)
        assert result.session_id == sid

    def test_returns_last_n_messages(self, store):
        """Should return only the last max_messages messages."""
        sid = str(uuid.uuid4())
        for i in range(15):
            store.save_message(sid, _make_message(sid, content=f"msg {i}"))

        result = store.get_context_window(sid, max_messages=10)
        assert len(result.messages) == 10

    def test_default_max_messages_is_10(self, store):
        """Default max_messages should be DEFAULT_MAX_MESSAGES (10)."""
        sid = str(uuid.uuid4())
        for i in range(15):
            store.save_message(sid, _make_message(sid, content=f"msg {i}"))

        result = store.get_context_window(sid)
        assert len(result.messages) == DEFAULT_MAX_MESSAGES

    def test_returns_all_messages_when_fewer_than_max(self, store):
        """If fewer messages exist than max_messages, all are returned."""
        sid = str(uuid.uuid4())
        for i in range(5):
            store.save_message(sid, _make_message(sid, content=f"msg {i}"))

        result = store.get_context_window(sid, max_messages=10)
        assert len(result.messages) == 5

    def test_messages_are_in_chronological_order(self, store):
        """Messages should be returned in the order they were saved."""
        sid = str(uuid.uuid4())
        contents = ["first", "second", "third"]
        for c in contents:
            store.save_message(sid, _make_message(sid, content=c))

        result = store.get_context_window(sid, max_messages=10)
        returned_contents = [m.content for m in result.messages]
        assert returned_contents == contents

    def test_total_tokens_reflects_all_messages(self, store):
        """total_tokens should be the sum of all stored message token counts."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid, token_count=10))
        store.save_message(sid, _make_message(sid, token_count=20))

        result = store.get_context_window(sid)
        assert result.total_tokens == 30

    def test_research_report_is_none_when_not_set(self, store):
        """research_report should be None when not initialised."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))
        result = store.get_context_window(sid)
        assert result.research_report is None

    def test_research_report_is_returned_when_set(self, store):
        """research_report should be returned when stored via init_session_context."""
        sid = str(uuid.uuid4())
        report = _make_report("Deep research answer")
        store.init_session_context(sid, research_report=report)
        store.save_message(sid, _make_message(sid))

        result = store.get_context_window(sid)
        assert result.research_report is not None
        assert result.research_report.answer == "Deep research answer"

    def test_sources_are_returned(self, store):
        """Sources stored via init_session_context should be returned."""
        sid = str(uuid.uuid4())
        sources = [_make_source(1), _make_source(2)]
        store.init_session_context(sid, sources=sources)
        store.save_message(sid, _make_message(sid))

        result = store.get_context_window(sid)
        assert len(result.sources) == 2
        assert result.sources[0].index == 1
        assert result.sources[1].index == 2

    def test_is_compressed_defaults_to_false(self, store):
        """is_compressed should be False for a fresh session."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))
        result = store.get_context_window(sid)
        assert result.is_compressed is False


# ---------------------------------------------------------------------------
# init_session_context tests
# ---------------------------------------------------------------------------


class TestInitSessionContext:
    def test_creates_context_hash(self, store, fake_redis):
        """init_session_context should create the Redis Hash."""
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        assert fake_redis.exists(_context_key(sid)) == 1

    def test_sets_ttl(self, store, fake_redis):
        """init_session_context should set the 24-hour TTL."""
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        ttl = fake_redis.ttl(_context_key(sid))
        assert SESSION_TTL_SECONDS - 5 <= ttl <= SESSION_TTL_SECONDS

    def test_stores_research_report(self, store):
        """Research report should be retrievable after init."""
        sid = str(uuid.uuid4())
        report = _make_report("Initial answer")
        store.init_session_context(sid, research_report=report)

        result = store.get_context_window(sid, max_messages=0)
        assert result.research_report is not None
        assert result.research_report.answer == "Initial answer"

    def test_stores_sources(self, store):
        """Sources should be retrievable after init."""
        sid = str(uuid.uuid4())
        sources = [_make_source(1)]
        store.init_session_context(sid, sources=sources)

        result = store.get_context_window(sid, max_messages=0)
        assert len(result.sources) == 1

    def test_registers_user_session(self, store, fake_redis):
        """When user_id is provided, session should be in user's sorted set."""
        sid = str(uuid.uuid4())
        uid = "user-init"
        store.init_session_context(sid, user_id=uid)

        members = fake_redis.zrange(_user_sessions_key(uid), 0, -1)
        decoded = [m.decode() if isinstance(m, bytes) else m for m in members]
        assert sid in decoded

    def test_empty_messages_after_init(self, store):
        """A freshly initialised session should have no messages."""
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        result = store.get_context_window(sid, max_messages=10)
        assert result.messages == []


# ---------------------------------------------------------------------------
# Redis key structure tests
# ---------------------------------------------------------------------------


class TestRedisKeyStructure:
    def test_context_key_format(self):
        """Context key should follow session:{session_id}:context pattern."""
        sid = "abc-123"
        assert _context_key(sid) == "session:abc-123:context"

    def test_user_sessions_key_format(self):
        """User sessions key should follow user:{user_id}:sessions pattern."""
        uid = "user-42"
        assert _user_sessions_key(uid) == "user:user-42:sessions"

    def test_different_sessions_have_different_keys(self):
        """Two different session IDs should produce different context keys."""
        sid1 = str(uuid.uuid4())
        sid2 = str(uuid.uuid4())
        assert _context_key(sid1) != _context_key(sid2)

    def test_different_users_have_different_keys(self):
        """Two different user IDs should produce different user session keys."""
        assert _user_sessions_key("user-A") != _user_sessions_key("user-B")


# ---------------------------------------------------------------------------
# Multi-user isolation tests
# ---------------------------------------------------------------------------


class TestMultiUserIsolation:
    def test_user_a_cannot_see_user_b_sessions(self, store, fake_redis):
        """Sessions registered for user-A should not appear in user-B's list."""
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())

        store.save_message(sid_a, _make_message(sid_a), user_id="user-A")
        store.save_message(sid_b, _make_message(sid_b), user_id="user-B")

        sessions_a = store.list_user_sessions("user-A")
        sessions_b = store.list_user_sessions("user-B")

        assert sid_a in sessions_a
        assert sid_b not in sessions_a
        assert sid_b in sessions_b
        assert sid_a not in sessions_b

    def test_context_keys_are_session_scoped_not_user_scoped(self, store, fake_redis):
        """Context data is keyed by session_id, not user_id."""
        sid = str(uuid.uuid4())
        # Two users saving to the same session_id would share context
        # (ownership enforcement is SessionManager's responsibility)
        store.save_message(sid, _make_message(sid, content="user-A msg"), user_id="user-A")
        store.save_message(sid, _make_message(sid, content="user-B msg"), user_id="user-B")

        result = store.get_context_window(sid, max_messages=10)
        assert len(result.messages) == 2

    def test_multiple_sessions_per_user(self, store, fake_redis):
        """A user can have multiple sessions in their sorted set."""
        uid = "user-multi"
        sids = [str(uuid.uuid4()) for _ in range(3)]
        for sid in sids:
            store.save_message(sid, _make_message(sid), user_id=uid)

        sessions = store.list_user_sessions(uid, limit=10)
        for sid in sids:
            assert sid in sessions


# ---------------------------------------------------------------------------
# delete_session_context tests
# ---------------------------------------------------------------------------


class TestDeleteSessionContext:
    def test_deletes_context_hash(self, store, fake_redis):
        """delete_session_context should remove the Redis Hash."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))
        store.delete_session_context(sid)
        assert fake_redis.exists(_context_key(sid)) == 0

    def test_removes_from_user_sorted_set(self, store, fake_redis):
        """delete_session_context should remove session from user's sorted set."""
        sid = str(uuid.uuid4())
        uid = "user-del"
        store.save_message(sid, _make_message(sid), user_id=uid)
        store.delete_session_context(sid, user_id=uid)

        members = fake_redis.zrange(_user_sessions_key(uid), 0, -1)
        decoded = [m.decode() if isinstance(m, bytes) else m for m in members]
        assert sid not in decoded

    def test_get_context_raises_after_delete(self, store):
        """get_context_window should raise after the context is deleted."""
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))
        store.delete_session_context(sid)

        with pytest.raises(SessionContextNotFoundError):
            store.get_context_window(sid)


# ---------------------------------------------------------------------------
# session_exists tests
# ---------------------------------------------------------------------------


class TestSessionExists:
    def test_returns_false_for_unknown_session(self, store):
        assert store.session_exists(str(uuid.uuid4())) is False

    def test_returns_true_after_save_message(self, store):
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))
        assert store.session_exists(sid) is True

    def test_returns_false_after_delete(self, store):
        sid = str(uuid.uuid4())
        store.save_message(sid, _make_message(sid))
        store.delete_session_context(sid)
        assert store.session_exists(sid) is False


# ---------------------------------------------------------------------------
# save_research_report tests
# ---------------------------------------------------------------------------


class TestSaveResearchReport:
    def test_report_is_retrievable(self, store):
        """Research report saved via save_research_report should be returned."""
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        report = _make_report("Updated answer")
        store.save_research_report(sid, report)

        result = store.get_context_window(sid, max_messages=0)
        assert result.research_report is not None
        assert result.research_report.answer == "Updated answer"

    def test_report_confidence_score_preserved(self, store):
        """confidence_score should be preserved through serialisation."""
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        report = ResearchReport(
            answer="Answer",
            sources=[],
            planner_decision={},
            confidence_score=0.75,
            review_feedback="OK",
        )
        store.save_research_report(sid, report)

        result = store.get_context_window(sid, max_messages=0)
        assert result.research_report.confidence_score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# update_sources tests
# ---------------------------------------------------------------------------


class TestUpdateSources:
    def test_sources_are_updated(self, store):
        """update_sources should replace the stored sources list."""
        sid = str(uuid.uuid4())
        store.init_session_context(sid, sources=[_make_source(1)])
        store.update_sources(sid, [_make_source(1), _make_source(2)])

        result = store.get_context_window(sid, max_messages=0)
        assert len(result.sources) == 2

    def test_source_fields_are_preserved(self, store):
        """Source fields should survive serialisation round-trip."""
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        source = Source(
            index=3,
            type="arxiv",
            title="Deep Learning Paper",
            url="https://arxiv.org/abs/1234.5678",
            arxiv_id="1234.5678",
            authors=["Alice", "Bob"],
            year=2023,
        )
        store.update_sources(sid, [source])

        result = store.get_context_window(sid, max_messages=0)
        s = result.sources[0]
        assert s.index == 3
        assert s.type == "arxiv"
        assert s.arxiv_id == "1234.5678"
        assert s.authors == ["Alice", "Bob"]
        assert s.year == 2023
