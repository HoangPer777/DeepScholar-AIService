"""
Tests for chat persistence bugfix and article_id fix.

This file covers:
  Task 1 — Bug condition exploration tests (FAIL on unfixed code, PASS after fix)
  Task 2 — Preservation property tests (PASS on both unfixed and fixed code)
  Task 6 — Integration tests (require fixes to be in place)

Bug 1a: GET /api/chat/history/{session_id} returned hardcoded [] instead of real Redis data
Bug 1b: docker-compose.yml lacked Redis service with persistent volume
Bug 2:  ChatResponse.article_id: int rejected None, causing HTTP 500

**Validates: Requirements 1.2, 1.4, 1.5, 2.2, 2.3, 2.5, 2.6, 3.1, 3.2, 3.6, 3.7**
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import fakeredis
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings as h_settings
from hypothesis import strategies as st

from app.core.memory_store import MemoryStore, SessionContextNotFoundError
from app.schemas.chat_models import Message, MessageRole


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _make_message(session_id: str, content: str = "Hello", role: MessageRole = MessageRole.USER) -> Message:
    return Message(session_id=session_id, role=role, content=content)


@pytest.fixture
def fake_redis():
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server)
    yield client
    client.flushall()
    client.close()


@pytest.fixture
def store(fake_redis):
    return MemoryStore(redis_client=fake_redis)


@pytest.fixture
def app_client():
    from app.main import app
    return TestClient(app)


# ===========================================================================
# TASK 1 — Bug Condition Exploration Tests
# These tests FAIL on unfixed code (confirming bugs exist)
# They PASS after the fix is applied
# ===========================================================================


class TestBugConditionExploration:
    """
    Bug condition exploration tests.

    On UNFIXED code: these tests FAIL — confirming the bugs exist.
    After fix: these tests PASS — confirming the bugs are resolved.

    **Validates: Requirements 1.2, 1.4, 1.5**
    """

    # -----------------------------------------------------------------------
    # Test 1a — Article ID None (Bug 2)
    # -----------------------------------------------------------------------

    def test_1a_chat_response_accepts_article_id_none(self):
        """
        Bug Condition: isBugCondition_ArticleIdNone(X) where X.article_id is None.

        UNFIXED: ChatResponse(article_id=None) raises ValidationError.
        FIXED:   ChatResponse(article_id=None) creates successfully.

        **Validates: Requirements 1.4, 1.5, 2.5**
        """
        from app.schemas.response import ChatResponse

        # This must NOT raise a ValidationError after the fix
        response = ChatResponse(
            session_id="test-session-x",
            article_id=None,
            answer="This is a follow-up answer after deep research.",
            confidence_score=0.9,
        )
        assert response.article_id is None
        assert response.answer == "This is a follow-up answer after deep research."
        assert response.session_id == "test-session-x"

    # -----------------------------------------------------------------------
    # Test 1b — History Hardcoded (Bug 1a)
    # -----------------------------------------------------------------------

    def test_1b_history_endpoint_returns_real_redis_data(self, fake_redis):
        """
        Bug Condition: isBugCondition_HistoryHardcoded(X) where session exists
        in Redis but endpoint returns hardcoded [].

        Setup: Insert 3 messages into Redis via MemoryStore.save_message().
        UNFIXED: GET /history/{session_id} returns {"history": []} — WRONG.
        FIXED:   GET /history/{session_id} returns {"history": [<3 messages>]}.

        **Validates: Requirements 1.2, 2.2**
        """
        from app.main import app

        session_id = str(uuid.uuid4())
        store = MemoryStore(redis_client=fake_redis)

        # Setup: init session and save 3 messages
        store.init_session_context(session_id)
        store.save_message(session_id, _make_message(session_id, "First question", MessageRole.USER))
        store.save_message(session_id, _make_message(session_id, "First answer", MessageRole.ASSISTANT))
        store.save_message(session_id, _make_message(session_id, "Second question", MessageRole.USER))

        # Verify data is in Redis
        assert store.session_exists(session_id)
        context = store.get_context_window(session_id)
        assert len(context.messages) == 3

        # Patch redis.from_url to return our fake_redis instance
        with patch("app.api.chatbot.redis_lib.from_url", return_value=fake_redis):
            client = TestClient(app)
            resp = client.get(f"/api/chat/history/{session_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id

        # FIXED: must return 3 real messages, not hardcoded []
        assert len(data["history"]) == 3, (
            f"Expected 3 messages but got {len(data['history'])}. "
            "Bug 1a: endpoint returned hardcoded [] instead of real Redis data."
        )
        contents = [m["content"] for m in data["history"]]
        assert "First question" in contents
        assert "First answer" in contents
        assert "Second question" in contents


# ===========================================================================
# TASK 2 — Preservation Property Tests
# These tests PASS on BOTH unfixed and fixed code (baseline behavior)
# ===========================================================================


class TestPreservationProperties:
    """
    Preservation property tests.

    These tests verify that existing correct behavior is preserved after the fix.
    They PASS on both unfixed and fixed code.

    **Validates: Requirements 3.1, 3.2, 3.6, 3.7**
    """

    # -----------------------------------------------------------------------
    # Test 2a — Integer article_id Preservation (PBT)
    # -----------------------------------------------------------------------

    @given(n=st.integers())
    @h_settings(max_examples=100)
    def test_2a_integer_article_id_preserved(self, n):
        """
        Property: For all integers n (including 0, negative, large),
        ChatResponse(article_id=n).article_id == n.

        Non-bug condition: X.article_id is not None.

        **Validates: Requirements 3.1, 2.6**
        """
        from app.schemas.response import ChatResponse

        response = ChatResponse(
            session_id="test-session",
            article_id=n,
            answer="Test answer",
            confidence_score=0.5,
        )
        assert response.article_id == n, (
            f"article_id round-trip failed: expected {n}, got {response.article_id}"
        )

    # -----------------------------------------------------------------------
    # Test 2b — History Empty Session Preservation
    # -----------------------------------------------------------------------

    def test_2b_history_empty_for_nonexistent_session(self, fake_redis):
        """
        Property: GET /history/{session_id} for a session that does NOT exist
        in Redis always returns {"history": []}.

        This behavior must be preserved after fix (empty list, not a crash).

        **Validates: Requirements 2.3, 3.2**
        """
        from app.main import app

        nonexistent_session_id = str(uuid.uuid4())

        # Verify session does NOT exist in Redis
        store = MemoryStore(redis_client=fake_redis)
        assert not store.session_exists(nonexistent_session_id)

        with patch("app.api.chatbot.redis_lib.from_url", return_value=fake_redis):
            client = TestClient(app)
            resp = client.get(f"/api/chat/history/{nonexistent_session_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == nonexistent_session_id
        assert data["history"] == [], (
            f"Expected empty history for nonexistent session, got: {data['history']}"
        )

    @given(session_id=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_")))
    @h_settings(max_examples=30, suppress_health_check=["function_scoped_fixture"])
    def test_2b_pbt_history_empty_for_random_nonexistent_sessions(self, session_id, fake_redis):
        """
        PBT: For random session IDs that don't exist in Redis,
        /history always returns {"history": []}.

        **Validates: Requirements 2.3**
        """
        from app.main import app

        store = MemoryStore(redis_client=fake_redis)
        # Ensure session does not exist
        if store.session_exists(session_id):
            store.delete_session_context(session_id)

        with patch("app.api.chatbot.redis_lib.from_url", return_value=fake_redis):
            client = TestClient(app)
            resp = client.get(f"/api/chat/history/{session_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["history"] == []

    # -----------------------------------------------------------------------
    # Test 2c — MemoryStore Save/Read Round-trip (PBT)
    # -----------------------------------------------------------------------

    @given(
        content=st.text(min_size=1, max_size=500),
        role=st.sampled_from([MessageRole.USER, MessageRole.ASSISTANT]),
    )
    @h_settings(max_examples=50, suppress_health_check=["function_scoped_fixture"])
    def test_2c_memory_store_save_read_roundtrip(self, content, role, fake_redis):
        """
        PBT: For random session IDs and message content,
        save_message + get_context_window returns the same messages.

        Non-bug condition: MemoryStore operations not involving /history endpoint.

        **Validates: Requirements 3.2, 3.6, 3.7**
        """
        store = MemoryStore(redis_client=fake_redis)
        session_id = str(uuid.uuid4())

        store.init_session_context(session_id)
        msg = Message(session_id=session_id, role=role, content=content)
        store.save_message(session_id, msg)

        context = store.get_context_window(session_id, max_messages=10)
        assert len(context.messages) == 1
        assert context.messages[0].content == content
        assert context.messages[0].role == role

        # Cleanup
        store.delete_session_context(session_id)

    def test_2c_memory_store_multiple_messages_order_preserved(self, store):
        """
        MemoryStore preserves message order and content for multiple messages.

        **Validates: Requirements 3.6, 3.7**
        """
        session_id = str(uuid.uuid4())
        store.init_session_context(session_id)

        messages = [
            ("Question 1", MessageRole.USER),
            ("Answer 1", MessageRole.ASSISTANT),
            ("Question 2", MessageRole.USER),
            ("Answer 2", MessageRole.ASSISTANT),
        ]

        for content, role in messages:
            store.save_message(session_id, _make_message(session_id, content, role))

        context = store.get_context_window(session_id, max_messages=10)
        assert len(context.messages) == 4
        for i, (content, role) in enumerate(messages):
            assert context.messages[i].content == content
            assert context.messages[i].role == role

    def test_2c_memory_store_ttl_refreshed_on_save(self, store, fake_redis):
        """
        MemoryStore refreshes TTL on every save_message call.

        **Validates: Requirements 3.2**
        """
        from app.core.memory_store import SESSION_TTL_SECONDS, _context_key

        session_id = str(uuid.uuid4())
        store.init_session_context(session_id)
        store.save_message(session_id, _make_message(session_id, "Hello"))

        ttl = fake_redis.ttl(_context_key(session_id))
        assert SESSION_TTL_SECONDS - 5 <= ttl <= SESSION_TTL_SECONDS


# ===========================================================================
# TASK 6 — Integration Tests
# These require fixes to be in place
# ===========================================================================


class TestIntegration:
    """
    Integration tests for the full chat flow after fixes.

    **Validates: Requirements 2.2, 2.5, 2.6, 3.1, 3.3, 3.4, 3.5**
    """

    def _mock_workflow_result(self, session_id: str = None) -> dict:
        return {
            "reviewed_answer": "This is the research answer.",
            "draft_answer": None,
            "confidence_score": 0.88,
            "review_feedback": "Good answer.",
            "need_clarification": False,
            "clarified_question": None,
            "external_context": [],
            "session_id": session_id,
        }

    # -----------------------------------------------------------------------
    # Test 6.1 — Full deep research → follow-up chat with article_id=None
    # -----------------------------------------------------------------------

    def test_6_1_chat_with_article_id_none_returns_200(self):
        """
        Integration: POST /api/chat/ with article_id=None returns HTTP 200.

        Simulates follow-up chat after deep research where no article is attached.

        **Validates: Requirements 2.5, 3.3, 3.4, 3.5**
        """
        from app.main import app

        session_id = str(uuid.uuid4())
        mock_result = self._mock_workflow_result(session_id=session_id)

        with patch("app.api.chatbot.run_chat_workflow", return_value=mock_result):
            client = TestClient(app)
            resp = client.post(
                "/api/chat/",
                json={
                    "question": "Can you explain the methodology in more detail?",
                    "article_id": None,
                    "session_id": session_id,
                },
            )

        assert resp.status_code == 200, (
            f"Expected HTTP 200 but got {resp.status_code}. "
            f"Response: {resp.text}"
        )
        data = resp.json()
        assert data["article_id"] is None
        assert data["answer"] == "This is the research answer."
        assert data["confidence_score"] == pytest.approx(0.88)

    # -----------------------------------------------------------------------
    # Test 6.2 — Chat with integer article_id → verify history persisted
    # -----------------------------------------------------------------------

    def test_6_2_chat_with_integer_article_id_and_history(self, fake_redis):
        """
        Integration: POST /api/chat/ with article_id=42 → GET /history/{session_id}
        returns the messages.

        **Validates: Requirements 2.2, 2.6, 3.1**
        """
        from app.main import app

        session_id = str(uuid.uuid4())
        mock_result = self._mock_workflow_result(session_id=session_id)

        # Step 1: POST /api/chat/ with article_id=42
        with patch("app.api.chatbot.run_chat_workflow", return_value=mock_result):
            client = TestClient(app)
            resp = client.post(
                "/api/chat/",
                json={
                    "question": "What is the main contribution of this paper?",
                    "article_id": 42,
                    "session_id": session_id,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["article_id"] == 42

        # Step 2: Manually save messages to fake_redis (simulating what the
        # workflow would do in a real scenario)
        store = MemoryStore(redis_client=fake_redis)
        store.init_session_context(session_id)
        store.save_message(
            session_id,
            _make_message(session_id, "What is the main contribution of this paper?", MessageRole.USER),
        )
        store.save_message(
            session_id,
            _make_message(session_id, "This is the research answer.", MessageRole.ASSISTANT),
        )

        # Step 3: GET /history/{session_id} → verify messages returned
        with patch("app.api.chatbot.redis_lib.from_url", return_value=fake_redis):
            resp2 = client.get(f"/api/chat/history/{session_id}")

        assert resp2.status_code == 200
        history_data = resp2.json()
        assert history_data["session_id"] == session_id
        assert len(history_data["history"]) == 2

        contents = [m["content"] for m in history_data["history"]]
        assert "What is the main contribution of this paper?" in contents
        assert "This is the research answer." in contents

    def test_6_2_article_id_42_preserved_in_response(self):
        """
        Preservation: article_id=42 in request → article_id=42 in response.

        **Validates: Requirements 2.6, 3.1**
        """
        from app.main import app

        session_id = str(uuid.uuid4())
        mock_result = self._mock_workflow_result(session_id=session_id)

        with patch("app.api.chatbot.run_chat_workflow", return_value=mock_result):
            client = TestClient(app)
            resp = client.post(
                "/api/chat/",
                json={
                    "question": "Explain the results section.",
                    "article_id": 42,
                    "session_id": session_id,
                },
            )

        assert resp.status_code == 200
        assert resp.json()["article_id"] == 42


# ===========================================================================
# Additional unit tests for MessageStore
# ===========================================================================


class TestMessageStore:
    """Unit tests for the new MessageStore class."""

    @pytest.fixture
    def db_and_store(self):
        """In-memory SQLite DB + MessageStore for unit testing."""
        import json
        from datetime import datetime

        from sqlalchemy import (
            Column,
            DateTime,
            ForeignKey,
            Integer,
            String,
            Text,
            create_engine,
        )
        from sqlalchemy.orm import declarative_base, relationship, sessionmaker
        from sqlalchemy.pool import StaticPool

        from app.core.message_store import MessageStore

        Base = declarative_base()

        class SessionORM(Base):
            __tablename__ = "sessions"
            session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(String(255), nullable=False)
            initial_query = Column(Text, nullable=False)
            created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
            updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
            status = Column(String(20), nullable=False, default="active")
            message_count = Column(Integer, nullable=False, default=0)
            messages = relationship("MessageORM", back_populates="session", cascade="all, delete-orphan")

        class MessageORM(Base):
            __tablename__ = "messages"
            message_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
            session_id = Column(String(36), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
            role = Column(String(20), nullable=False)
            content = Column(Text, nullable=False)
            timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
            token_count = Column(Integer, nullable=False, default=0)
            # Use String for SQLite compatibility (production uses JSONB)
            msg_metadata = Column("metadata", String, nullable=True)
            session = relationship("SessionORM", back_populates="messages")

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        # Patch the ORM classes used by MessageStore
        import app.core.message_store as ms_module
        original_session_orm = ms_module.SessionORM
        original_message_orm = ms_module.MessageORM
        ms_module.SessionORM = SessionORM
        ms_module.MessageORM = MessageORM

        yield db, MessageStore(), SessionORM, MessageORM

        ms_module.SessionORM = original_session_orm
        ms_module.MessageORM = original_message_orm
        db.close()

    def test_message_store_save_and_get(self, db_and_store):
        """save_message + get_messages round-trip works correctly."""
        db, store, SessionORM, MessageORM = db_and_store

        session_id = str(uuid.uuid4())
        # Create session first
        db_session = SessionORM(
            session_id=session_id,
            user_id="user-1",
            initial_query="Test query",
        )
        db.add(db_session)
        db.commit()

        # Save messages — pass metadata as None to avoid SQLite dict binding issue
        store.save_message(db, session_id, "user", "Hello world", metadata=None)
        store.save_message(db, session_id, "assistant", "Hi there!", metadata=None)

        # Retrieve messages
        messages = store.get_messages(db, session_id)
        assert len(messages) == 2
        assert messages[0].content == "Hello world"
        assert messages[0].role == "user"
        assert messages[1].content == "Hi there!"
        assert messages[1].role == "assistant"

    def test_message_store_empty_for_nonexistent_session(self, db_and_store):
        """get_messages returns empty list for nonexistent session."""
        db, store, _, _ = db_and_store
        messages = store.get_messages(db, str(uuid.uuid4()))
        assert messages == []


# ===========================================================================
# Docker-compose validation test
# ===========================================================================


class TestDockerComposeConfig:
    """Validate docker-compose.yml has Redis service with persistent volume.

    **Validates: Requirements 1.3, 2.4**
    """

    def test_docker_compose_has_redis_service(self):
        """docker-compose.yml must define a redis service."""
        import os
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(compose_path) as f:
            config = yaml.safe_load(f)

        assert "redis" in config.get("services", {}), (
            "Bug 1b: docker-compose.yml missing 'redis' service"
        )

    def test_docker_compose_redis_has_persistent_volume(self):
        """Redis service must have a named volume for persistence."""
        import os
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(compose_path) as f:
            config = yaml.safe_load(f)

        redis_service = config.get("services", {}).get("redis", {})
        volumes = redis_service.get("volumes", [])
        assert len(volumes) > 0, "Redis service must have at least one volume"

        # Check top-level volumes declaration
        assert "volumes" in config, "docker-compose.yml must declare top-level volumes"
        assert "redis_data" in config["volumes"], "redis_data volume must be declared"

    def test_docker_compose_redis_has_appendonly(self):
        """Redis service must use AOF persistence (--appendonly yes)."""
        import os
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(compose_path) as f:
            config = yaml.safe_load(f)

        redis_service = config.get("services", {}).get("redis", {})
        command = redis_service.get("command", "")
        assert "appendonly yes" in command, (
            "Redis service must use AOF persistence: 'redis-server --appendonly yes'"
        )

    def test_docker_compose_ai_service_uses_internal_redis_url(self):
        """ai-service must connect to Redis via internal Docker network."""
        import os
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(compose_path) as f:
            config = yaml.safe_load(f)

        ai_service = config.get("services", {}).get("ai-service", {})
        env = ai_service.get("environment", [])

        # env can be a list of "KEY=VALUE" strings or a dict
        if isinstance(env, list):
            redis_url_entries = [e for e in env if "REDIS_URL" in str(e)]
            assert len(redis_url_entries) > 0, "ai-service must set REDIS_URL"
            redis_url = redis_url_entries[0]
            assert "host.docker.internal" not in str(redis_url), (
                "REDIS_URL must use internal Docker network (redis://redis:...), "
                "not host.docker.internal"
            )
            assert "redis://redis:" in str(redis_url), (
                "REDIS_URL must point to the internal redis service"
            )
        elif isinstance(env, dict):
            assert "REDIS_URL" in env
            assert "host.docker.internal" not in env["REDIS_URL"]
            assert "redis://redis:" in env["REDIS_URL"]

    def test_docker_compose_ai_service_depends_on_redis(self):
        """ai-service must declare depends_on redis."""
        import os
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(compose_path) as f:
            config = yaml.safe_load(f)

        ai_service = config.get("services", {}).get("ai-service", {})
        depends_on = ai_service.get("depends_on", [])
        assert "redis" in depends_on, (
            "ai-service must declare depends_on: [redis]"
        )
