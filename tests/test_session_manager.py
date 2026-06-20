"""
Unit tests for SessionManager CRUD operations.

Tests use an in-memory SQLite database so they run without a live PostgreSQL
instance.  SQLite does not support PostgreSQL-specific types (UUID, JSONB) so
we patch the column types via a lightweight fixture that creates the schema
from the ORM models.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 12.1, 12.2
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, String, event
from sqlalchemy.orm import sessionmaker, Session as DBSession
from sqlalchemy.pool import StaticPool

from app.core.session_manager import (
    MAX_ACTIVE_SESSIONS_PER_USER,
    ConcurrentSessionLimitError,
    SessionManager,
    SessionNotFoundError,
    SessionOwnershipError,
)
from app.schemas.chat_models import Session, SessionMetadata, SessionStatus


# ---------------------------------------------------------------------------
# In-memory SQLite test database
# ---------------------------------------------------------------------------

# We need to override UUID columns with String for SQLite compatibility.
# The simplest approach: create a fresh Base + patched models for tests.

from sqlalchemy import Column, Text, DateTime, Integer, Index, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

TestBase = declarative_base()


class SessionORM(TestBase):
    """SQLite-compatible version of the Session ORM model for testing."""

    __tablename__ = "sessions"

    session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    initial_query = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    status = Column(String(20), nullable=False, default="active")
    message_count = Column(Integer, nullable=False, default=0)

    messages = relationship(
        "MessageORM",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Session(session_id={self.session_id}, user_id={self.user_id})>"


class MessageORM(TestBase):
    """SQLite-compatible version of the Message ORM model for testing."""

    __tablename__ = "messages"

    message_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)

    session = relationship("SessionORM", back_populates="messages")


@pytest.fixture(scope="function")
def db() -> DBSession:
    """Provide a fresh in-memory SQLite session for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestBase.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestSessionLocal()
    yield session
    session.close()
    TestBase.metadata.drop_all(engine)


@pytest.fixture
def manager() -> SessionManager:
    """Return a SessionManager configured to use the SQLite-compatible test ORM."""
    return SessionManager(session_orm_class=SessionORM)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _create_raw_session(db: DBSession, user_id: str, query: str = "test query", status: str = "active") -> SessionORM:
    """Directly insert a session row, bypassing SessionManager."""
    s = SessionORM(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        initial_query=query,
        status=status,
        message_count=0,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ---------------------------------------------------------------------------
# create_session tests
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_returns_session_pydantic_model(self, manager, db):
        result = manager.create_session(db, user_id="user-1", initial_query="What is AI?")
        assert isinstance(result, Session)

    def test_session_id_is_valid_uuid(self, manager, db):
        result = manager.create_session(db, user_id="user-1", initial_query="What is AI?")
        parsed = uuid.UUID(result.session_id)
        assert str(parsed) == result.session_id

    def test_each_session_gets_unique_id(self, manager, db):
        s1 = manager.create_session(db, user_id="user-1", initial_query="Query A")
        s2 = manager.create_session(db, user_id="user-1", initial_query="Query B")
        assert s1.session_id != s2.session_id

    def test_stores_user_id(self, manager, db):
        result = manager.create_session(db, user_id="user-42", initial_query="Test")
        assert result.user_id == "user-42"

    def test_stores_initial_query(self, manager, db):
        query = "Tell me about quantum computing"
        result = manager.create_session(db, user_id="user-1", initial_query=query)
        assert result.initial_query == query

    def test_default_status_is_active(self, manager, db):
        result = manager.create_session(db, user_id="user-1", initial_query="Test")
        assert result.status == SessionStatus.ACTIVE

    def test_default_message_count_is_zero(self, manager, db):
        result = manager.create_session(db, user_id="user-1", initial_query="Test")
        assert result.message_count == 0

    def test_session_persisted_to_database(self, manager, db):
        result = manager.create_session(db, user_id="user-1", initial_query="Test")
        row = db.query(SessionORM).filter(SessionORM.session_id == result.session_id).first()
        assert row is not None
        assert row.user_id == "user-1"

    def test_raises_when_active_session_limit_reached(self, manager, db):
        # Create MAX_ACTIVE_SESSIONS_PER_USER active sessions
        for i in range(MAX_ACTIVE_SESSIONS_PER_USER):
            _create_raw_session(db, user_id="user-limit", query=f"Query {i}", status="active")

        with pytest.raises(ConcurrentSessionLimitError):
            manager.create_session(db, user_id="user-limit", initial_query="One too many")

    def test_inactive_sessions_do_not_count_toward_limit(self, manager, db):
        # Fill up with inactive sessions — should not block creation
        for i in range(MAX_ACTIVE_SESSIONS_PER_USER):
            _create_raw_session(db, user_id="user-inactive", query=f"Query {i}", status="inactive")

        # Should succeed because none are active
        result = manager.create_session(db, user_id="user-inactive", initial_query="New active")
        assert result.status == SessionStatus.ACTIVE

    def test_different_users_have_independent_limits(self, manager, db):
        # Fill user-A's limit
        for i in range(MAX_ACTIVE_SESSIONS_PER_USER):
            _create_raw_session(db, user_id="user-A", query=f"Query {i}", status="active")

        # user-B should still be able to create a session
        result = manager.create_session(db, user_id="user-B", initial_query="User B query")
        assert result.user_id == "user-B"


# ---------------------------------------------------------------------------
# get_session tests
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_returns_correct_session(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1", query="My query")
        result = manager.get_session(db, session_id=raw.session_id, user_id="user-1")
        assert result.session_id == raw.session_id
        assert result.initial_query == "My query"

    def test_raises_session_not_found_for_unknown_id(self, manager, db):
        with pytest.raises(SessionNotFoundError):
            manager.get_session(db, session_id=str(uuid.uuid4()), user_id="user-1")

    def test_raises_session_not_found_for_invalid_uuid(self, manager, db):
        with pytest.raises(SessionNotFoundError):
            manager.get_session(db, session_id="not-a-uuid", user_id="user-1")

    def test_raises_ownership_error_for_wrong_user(self, manager, db):
        raw = _create_raw_session(db, user_id="user-owner", query="Secret query")
        with pytest.raises(SessionOwnershipError):
            manager.get_session(db, session_id=raw.session_id, user_id="user-attacker")

    def test_owner_can_access_own_session(self, manager, db):
        raw = _create_raw_session(db, user_id="user-owner", query="My research")
        result = manager.get_session(db, session_id=raw.session_id, user_id="user-owner")
        assert result.user_id == "user-owner"

    def test_returns_pydantic_session_model(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1")
        result = manager.get_session(db, session_id=raw.session_id, user_id="user-1")
        assert isinstance(result, Session)


# ---------------------------------------------------------------------------
# list_sessions tests
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_returns_only_user_sessions(self, manager, db):
        _create_raw_session(db, user_id="user-A", query="A query")
        _create_raw_session(db, user_id="user-B", query="B query")

        results = manager.list_sessions(db, user_id="user-A")
        assert all(r.session_id is not None for r in results)
        # Verify by checking the raw DB that only user-A sessions are returned
        user_a_ids = {
            str(s.session_id)
            for s in db.query(SessionORM).filter(SessionORM.user_id == "user-A").all()
        }
        returned_ids = {r.session_id for r in results}
        assert returned_ids == user_a_ids

    def test_returns_empty_list_for_user_with_no_sessions(self, manager, db):
        results = manager.list_sessions(db, user_id="user-nobody")
        assert results == []

    def test_returns_session_metadata_objects(self, manager, db):
        _create_raw_session(db, user_id="user-1")
        results = manager.list_sessions(db, user_id="user-1")
        assert len(results) >= 1
        assert isinstance(results[0], SessionMetadata)

    def test_pagination_limit(self, manager, db):
        for i in range(5):
            _create_raw_session(db, user_id="user-page", query=f"Query {i}")

        results = manager.list_sessions(db, user_id="user-page", limit=3)
        assert len(results) == 3

    def test_pagination_offset(self, manager, db):
        for i in range(5):
            _create_raw_session(db, user_id="user-page2", query=f"Query {i}")

        all_results = manager.list_sessions(db, user_id="user-page2", limit=5)
        offset_results = manager.list_sessions(db, user_id="user-page2", limit=5, offset=2)
        assert len(offset_results) == 3
        # The offset results should not overlap with the first 2
        all_ids = [r.session_id for r in all_results]
        offset_ids = [r.session_id for r in offset_results]
        assert offset_ids == all_ids[2:]

    def test_filter_by_status_active(self, manager, db):
        _create_raw_session(db, user_id="user-filter", status="active")
        _create_raw_session(db, user_id="user-filter", status="inactive")
        _create_raw_session(db, user_id="user-filter", status="archived")

        results = manager.list_sessions(db, user_id="user-filter", status=SessionStatus.ACTIVE)
        assert all(r.status == SessionStatus.ACTIVE for r in results)
        assert len(results) == 1

    def test_filter_by_status_inactive(self, manager, db):
        _create_raw_session(db, user_id="user-filter2", status="active")
        _create_raw_session(db, user_id="user-filter2", status="inactive")

        results = manager.list_sessions(db, user_id="user-filter2", status=SessionStatus.INACTIVE)
        assert len(results) == 1
        assert results[0].status == SessionStatus.INACTIVE

    def test_no_status_filter_returns_all(self, manager, db):
        _create_raw_session(db, user_id="user-all", status="active")
        _create_raw_session(db, user_id="user-all", status="inactive")

        results = manager.list_sessions(db, user_id="user-all")
        assert len(results) == 2

    def test_ordered_by_created_at_descending(self, manager, db):
        """Newest sessions should appear first."""
        for i in range(3):
            _create_raw_session(db, user_id="user-order", query=f"Query {i}")

        results = manager.list_sessions(db, user_id="user-order")
        timestamps = [r.created_at for r in results]
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# update_session_status tests
# ---------------------------------------------------------------------------


class TestUpdateSessionStatus:
    def test_updates_status_to_inactive(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1", status="active")
        result = manager.update_session_status(
            db, session_id=raw.session_id, status=SessionStatus.INACTIVE
        )
        assert result.status == SessionStatus.INACTIVE

    def test_updates_status_to_archived(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1", status="active")
        result = manager.update_session_status(
            db, session_id=raw.session_id, status=SessionStatus.ARCHIVED
        )
        assert result.status == SessionStatus.ARCHIVED

    def test_updates_status_back_to_active(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1", status="inactive")
        result = manager.update_session_status(
            db, session_id=raw.session_id, status=SessionStatus.ACTIVE
        )
        assert result.status == SessionStatus.ACTIVE

    def test_persists_status_change_to_database(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1", status="active")
        manager.update_session_status(
            db, session_id=raw.session_id, status=SessionStatus.INACTIVE
        )
        db.expire_all()
        row = db.query(SessionORM).filter(SessionORM.session_id == raw.session_id).first()
        assert row.status == "inactive"

    def test_raises_session_not_found_for_unknown_id(self, manager, db):
        with pytest.raises(SessionNotFoundError):
            manager.update_session_status(
                db, session_id=str(uuid.uuid4()), status=SessionStatus.INACTIVE
            )

    def test_raises_ownership_error_when_user_id_provided_and_wrong(self, manager, db):
        raw = _create_raw_session(db, user_id="user-owner")
        with pytest.raises(SessionOwnershipError):
            manager.update_session_status(
                db,
                session_id=raw.session_id,
                status=SessionStatus.INACTIVE,
                user_id="user-attacker",
            )

    def test_no_ownership_check_when_user_id_is_none(self, manager, db):
        """Background tasks can update status without user_id."""
        raw = _create_raw_session(db, user_id="user-owner")
        result = manager.update_session_status(
            db,
            session_id=raw.session_id,
            status=SessionStatus.INACTIVE,
            user_id=None,
        )
        assert result.status == SessionStatus.INACTIVE

    def test_returns_pydantic_session_model(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1")
        result = manager.update_session_status(
            db, session_id=raw.session_id, status=SessionStatus.INACTIVE
        )
        assert isinstance(result, Session)


# ---------------------------------------------------------------------------
# delete_session tests
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_returns_true_on_success(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1")
        result = manager.delete_session(db, session_id=raw.session_id, user_id="user-1")
        assert result is True

    def test_session_removed_from_database(self, manager, db):
        raw = _create_raw_session(db, user_id="user-1")
        manager.delete_session(db, session_id=raw.session_id, user_id="user-1")
        row = db.query(SessionORM).filter(SessionORM.session_id == raw.session_id).first()
        assert row is None

    def test_raises_session_not_found_for_unknown_id(self, manager, db):
        with pytest.raises(SessionNotFoundError):
            manager.delete_session(db, session_id=str(uuid.uuid4()), user_id="user-1")

    def test_raises_ownership_error_for_wrong_user(self, manager, db):
        raw = _create_raw_session(db, user_id="user-owner")
        with pytest.raises(SessionOwnershipError):
            manager.delete_session(db, session_id=raw.session_id, user_id="user-attacker")

    def test_session_not_deleted_on_ownership_error(self, manager, db):
        raw = _create_raw_session(db, user_id="user-owner")
        with pytest.raises(SessionOwnershipError):
            manager.delete_session(db, session_id=raw.session_id, user_id="user-attacker")
        # Session should still exist
        row = db.query(SessionORM).filter(SessionORM.session_id == raw.session_id).first()
        assert row is not None

    def test_cascade_deletes_messages(self, manager, db):
        """Deleting a session should cascade-delete its messages."""
        raw = _create_raw_session(db, user_id="user-1")
        # Add a message directly
        msg = MessageORM(
            message_id=str(uuid.uuid4()),
            session_id=raw.session_id,
            role="user",
            content="Hello",
            token_count=5,
        )
        db.add(msg)
        db.commit()

        manager.delete_session(db, session_id=raw.session_id, user_id="user-1")

        remaining = db.query(MessageORM).filter(MessageORM.session_id == raw.session_id).all()
        assert remaining == []

    def test_deleting_one_session_does_not_affect_others(self, manager, db):
        raw1 = _create_raw_session(db, user_id="user-1", query="Query 1")
        raw2 = _create_raw_session(db, user_id="user-1", query="Query 2")

        manager.delete_session(db, session_id=raw1.session_id, user_id="user-1")

        row2 = db.query(SessionORM).filter(SessionORM.session_id == raw2.session_id).first()
        assert row2 is not None

    def test_raises_session_not_found_for_invalid_uuid(self, manager, db):
        with pytest.raises(SessionNotFoundError):
            manager.delete_session(db, session_id="bad-uuid", user_id="user-1")
