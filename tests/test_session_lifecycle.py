"""
Unit tests for session lifecycle management.

Tests cover:
- mark_inactive_sessions: sessions idle ≥ 24 h are marked inactive
- increment_message_count: message_count increments and updated_at refreshes

All tests use an in-memory SQLite database so they run without a live
PostgreSQL instance.  The SQLite-compatible ORM models mirror the production
schema but use String columns instead of UUID/JSONB.

Requirements: 1.7, 1.8
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session as DBSession
from sqlalchemy.pool import StaticPool

from app.core.session_lifecycle import (
    INACTIVITY_THRESHOLD_HOURS,
    increment_message_count,
    mark_inactive_sessions,
)

# ---------------------------------------------------------------------------
# SQLite-compatible ORM models for testing
# ---------------------------------------------------------------------------

TestBase = declarative_base()


class SessionORM(TestBase):
    """SQLite-compatible session model (no UUID/JSONB columns)."""

    __tablename__ = "sessions"

    session_id = Column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id = Column(String(255), nullable=False, index=True)
    initial_query = Column(Text, nullable=False, default="test query")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    status = Column(String(20), nullable=False, default="active")
    message_count = Column(Integer, nullable=False, default=0)

    messages = relationship(
        "MessageORM",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class MessageORM(TestBase):
    """SQLite-compatible message model."""

    __tablename__ = "messages"

    message_id = Column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id = Column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String(20), nullable=False, default="user")
    content = Column(Text, nullable=False, default="")
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    token_count = Column(Integer, nullable=False, default=0)

    session = relationship("SessionORM", back_populates="messages")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    db: DBSession,
    *,
    user_id: str = "user-1",
    status: str = "active",
    hours_since_update: float = 0.0,
    query: str = "test query",
) -> SessionORM:
    """Insert a session row with a controlled ``updated_at`` timestamp."""
    updated_at = datetime.now(timezone.utc) - timedelta(hours=hours_since_update)
    row = SessionORM(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        initial_query=query,
        status=status,
        message_count=0,
        updated_at=updated_at,
        created_at=updated_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# ---------------------------------------------------------------------------
# mark_inactive_sessions tests
# ---------------------------------------------------------------------------


class TestMarkInactiveSessions:
    """Tests for mark_inactive_sessions()."""

    def test_returns_zero_when_no_sessions(self, db):
        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 0

    def test_active_session_idle_over_24h_is_marked_inactive(self, db):
        row = _make_session(db, status="active", hours_since_update=25)
        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 1
        db.expire_all()
        updated = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()
        assert updated.status == "inactive"

    def test_active_session_idle_exactly_24h_is_marked_inactive(self, db):
        """Boundary: exactly 24 h idle should be caught (updated_at < cutoff)."""
        # Use 24 h + 1 second to be safely past the boundary
        row = _make_session(db, status="active", hours_since_update=24 + (1 / 3600))
        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 1
        db.expire_all()
        updated = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()
        assert updated.status == "inactive"

    def test_active_session_idle_under_24h_is_not_changed(self, db):
        row = _make_session(db, status="active", hours_since_update=10)
        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 0
        db.expire_all()
        updated = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()
        assert updated.status == "active"

    def test_already_inactive_session_is_not_double_counted(self, db):
        _make_session(db, status="inactive", hours_since_update=30)
        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 0

    def test_archived_session_is_not_affected(self, db):
        _make_session(db, status="archived", hours_since_update=30)
        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 0

    def test_multiple_stale_sessions_all_marked(self, db):
        for i in range(5):
            _make_session(db, user_id=f"user-{i}", status="active", hours_since_update=48)
        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 5

    def test_only_stale_sessions_are_marked(self, db):
        stale = _make_session(db, status="active", hours_since_update=30)
        fresh = _make_session(db, status="active", hours_since_update=1)

        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 1

        db.expire_all()
        assert (
            db.query(SessionORM)
            .filter(SessionORM.session_id == stale.session_id)
            .first()
            .status
            == "inactive"
        )
        assert (
            db.query(SessionORM)
            .filter(SessionORM.session_id == fresh.session_id)
            .first()
            .status
            == "active"
        )

    def test_returns_correct_count_for_mixed_sessions(self, db):
        # 3 stale active, 2 fresh active, 1 stale inactive
        for _ in range(3):
            _make_session(db, status="active", hours_since_update=48)
        for _ in range(2):
            _make_session(db, status="active", hours_since_update=2)
        _make_session(db, status="inactive", hours_since_update=48)

        count = mark_inactive_sessions(db, orm_class=SessionORM)
        assert count == 3

    def test_idempotent_second_call_returns_zero(self, db):
        _make_session(db, status="active", hours_since_update=30)
        first = mark_inactive_sessions(db, orm_class=SessionORM)
        second = mark_inactive_sessions(db, orm_class=SessionORM)
        assert first == 1
        assert second == 0

    def test_threshold_constant_is_24(self):
        assert INACTIVITY_THRESHOLD_HOURS == 24


# ---------------------------------------------------------------------------
# increment_message_count tests
# ---------------------------------------------------------------------------


class TestIncrementMessageCount:
    """Tests for increment_message_count()."""

    def test_returns_true_for_existing_session(self, db):
        row = _make_session(db)
        result = increment_message_count(db, row.session_id, orm_class=SessionORM)
        assert result is True

    def test_returns_false_for_nonexistent_session(self, db):
        result = increment_message_count(
            db, str(uuid.uuid4()), orm_class=SessionORM
        )
        assert result is False

    def test_increments_message_count_by_one(self, db):
        row = _make_session(db)
        assert row.message_count == 0
        increment_message_count(db, row.session_id, orm_class=SessionORM)
        db.expire_all()
        updated = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()
        assert updated.message_count == 1

    def test_multiple_increments_accumulate(self, db):
        row = _make_session(db)
        for _ in range(5):
            increment_message_count(db, row.session_id, orm_class=SessionORM)
        db.expire_all()
        updated = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()
        assert updated.message_count == 5

    def test_updated_at_is_refreshed(self, db):
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        row = _make_session(db, hours_since_update=2)
        original_updated_at = row.updated_at

        increment_message_count(db, row.session_id, orm_class=SessionORM)
        db.expire_all()
        updated = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()

        # updated_at should be more recent than the original
        # Compare as naive datetimes if needed (SQLite strips tzinfo)
        new_updated_at = updated.updated_at
        if new_updated_at.tzinfo is None:
            original_naive = original_updated_at.replace(tzinfo=None)
        else:
            original_naive = original_updated_at

        assert new_updated_at > original_naive

    def test_increment_does_not_affect_other_sessions(self, db):
        row1 = _make_session(db, user_id="user-1")
        row2 = _make_session(db, user_id="user-2")

        increment_message_count(db, row1.session_id, orm_class=SessionORM)
        db.expire_all()

        r2 = db.query(SessionORM).filter(SessionORM.session_id == row2.session_id).first()
        assert r2.message_count == 0

    def test_increment_works_on_inactive_session(self, db):
        """Metadata should still be trackable even for inactive sessions."""
        row = _make_session(db, status="inactive")
        result = increment_message_count(db, row.session_id, orm_class=SessionORM)
        assert result is True
        db.expire_all()
        updated = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()
        assert updated.message_count == 1

    def test_increment_persists_to_database(self, db):
        """Verify the change is committed and visible after expire_all."""
        row = _make_session(db)
        increment_message_count(db, row.session_id, orm_class=SessionORM)
        db.expire_all()
        fresh = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()
        assert fresh.message_count == 1

    def test_increment_after_mark_inactive_updates_count(self, db):
        """Lifecycle functions can be composed: mark inactive then increment."""
        row = _make_session(db, status="active", hours_since_update=30)
        mark_inactive_sessions(db, orm_class=SessionORM)
        db.expire_all()

        # Session is now inactive; incrementing should still work
        result = increment_message_count(db, row.session_id, orm_class=SessionORM)
        assert result is True
        db.expire_all()
        updated = db.query(SessionORM).filter(SessionORM.session_id == row.session_id).first()
        assert updated.message_count == 1
        assert updated.status == "inactive"
