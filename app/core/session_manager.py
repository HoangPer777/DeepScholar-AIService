"""
SessionManager: CRUD operations for conversation sessions.

This module provides the SessionManager class which handles the full lifecycle
of conversation sessions for the Deep Research Memory Chatbot feature.

All database operations use SQLAlchemy ORM sessions and enforce user ownership
to satisfy multi-user isolation requirements.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 12.1, 12.2
"""

from __future__ import annotations

import uuid
import logging
from typing import List, Optional, Type

from sqlalchemy.orm import Session as DBSession
from sqlalchemy.exc import SQLAlchemyError

from app.core.session_models import Session as SessionORM
from app.schemas.chat_models import (
    Session,
    SessionMetadata,
    SessionStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class SessionNotFoundError(Exception):
    """Raised when a session does not exist or does not belong to the user."""


class SessionOwnershipError(Exception):
    """Raised when a user attempts to access another user's session."""


class ConcurrentSessionLimitError(Exception):
    """Raised when a user exceeds the maximum number of active sessions."""


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

MAX_ACTIVE_SESSIONS_PER_USER = 10


class SessionManager:
    """Manages conversation session lifecycle with full CRUD support.

    All methods accept a SQLAlchemy ``db`` session so that callers control
    transaction boundaries (useful for testing and for composing operations).

    Multi-user isolation is enforced at the application layer: every read and
    write operation validates that the requesting ``user_id`` matches the
    session owner.  Row-level security in PostgreSQL provides a second layer
    of defence (see migration 001).

    Args:
        session_orm_class: The SQLAlchemy ORM class to use for session queries.
            Defaults to the production ``SessionORM``.  Pass a test-compatible
            class (e.g. a SQLite-friendly model) in unit tests.

    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 12.1, 12.2
    """

    def __init__(self, session_orm_class: Optional[Type] = None):
        # Allow injection of a test-compatible ORM class
        self._ORM = session_orm_class if session_orm_class is not None else SessionORM

    # ------------------------------------------------------------------
    # create_session
    # ------------------------------------------------------------------

    def create_session(
        self,
        db: DBSession,
        user_id: str,
        initial_query: str,
    ) -> Session:
        """Create a new conversation session for *user_id*.

        Generates a UUID for the session, persists the record to PostgreSQL,
        and returns a Pydantic ``Session`` object.

        Args:
            db:            SQLAlchemy database session.
            user_id:       Identifier of the authenticated user.
            initial_query: The research query that starts this session.

        Returns:
            A ``Session`` Pydantic model representing the newly created session.

        Raises:
            ConcurrentSessionLimitError: If the user already has
                ``MAX_ACTIVE_SESSIONS_PER_USER`` active sessions.
            SQLAlchemyError: On unexpected database errors.

        Requirements: 1.1, 1.2, 1.3, 12.1
        """
        ORM = self._ORM

        # Enforce concurrent session limit (Requirement 1.8 / design note)
        active_count = (
            db.query(ORM)
            .filter(
                ORM.user_id == user_id,
                ORM.status == SessionStatus.ACTIVE.value,
            )
            .count()
        )
        if active_count >= MAX_ACTIVE_SESSIONS_PER_USER:
            raise ConcurrentSessionLimitError(
                f"User '{user_id}' already has {active_count} active sessions "
                f"(limit: {MAX_ACTIVE_SESSIONS_PER_USER})."
            )

        session_id = str(uuid.uuid4())

        db_session = ORM(
            session_id=session_id,
            user_id=user_id,
            initial_query=initial_query,
            status=SessionStatus.ACTIVE.value,
            message_count=0,
        )

        try:
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("Failed to create session for user '%s': %s", user_id, exc)
            raise

        logger.info(
            "Created session %s for user '%s'", db_session.session_id, user_id
        )
        return self._orm_to_pydantic(db_session)

    # ------------------------------------------------------------------
    # get_session
    # ------------------------------------------------------------------

    def get_session(
        self,
        db: DBSession,
        session_id: str,
        user_id: str,
    ) -> Session:
        """Retrieve a session by ID, validating that it belongs to *user_id*.

        Args:
            db:         SQLAlchemy database session.
            session_id: UUID string of the session to retrieve.
            user_id:    Identifier of the authenticated user.

        Returns:
            A ``Session`` Pydantic model.

        Raises:
            SessionNotFoundError:  If no session with *session_id* exists.
            SessionOwnershipError: If the session exists but belongs to a
                different user.

        Requirements: 1.4, 1.5, 12.1, 12.2
        """
        db_session = self._fetch_session_or_raise(db, session_id)

        if db_session.user_id != user_id:
            logger.warning(
                "Ownership violation: user '%s' attempted to access session %s "
                "(owned by '%s')",
                user_id,
                session_id,
                db_session.user_id,
            )
            raise SessionOwnershipError(
                f"Session '{session_id}' does not belong to user '{user_id}'."
            )

        return self._orm_to_pydantic(db_session)

    # ------------------------------------------------------------------
    # list_sessions
    # ------------------------------------------------------------------

    def list_sessions(
        self,
        db: DBSession,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        status: Optional[SessionStatus] = None,
    ) -> List[SessionMetadata]:
        """List sessions belonging to *user_id* with optional pagination.

        Results are ordered by ``created_at`` descending (newest first).

        Args:
            db:      SQLAlchemy database session.
            user_id: Identifier of the authenticated user.
            limit:   Maximum number of sessions to return (default 50).
            offset:  Number of sessions to skip for pagination (default 0).
            status:  Optional filter by session status.

        Returns:
            A list of ``SessionMetadata`` Pydantic models.

        Requirements: 1.8, 12.1
        """
        ORM = self._ORM
        query = db.query(ORM).filter(ORM.user_id == user_id)

        if status is not None:
            query = query.filter(ORM.status == status.value)

        db_sessions = (
            query.order_by(ORM.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return [self._orm_to_metadata(s) for s in db_sessions]

    # ------------------------------------------------------------------
    # update_session_status
    # ------------------------------------------------------------------

    def update_session_status(
        self,
        db: DBSession,
        session_id: str,
        status: SessionStatus,
        user_id: Optional[str] = None,
    ) -> Session:
        """Update the status of a session.

        If *user_id* is provided the ownership is validated before the update.
        Passing ``user_id=None`` is intended for internal/background tasks
        (e.g. the inactivity background job) that do not have a user context.

        Args:
            db:         SQLAlchemy database session.
            session_id: UUID string of the session to update.
            status:     New ``SessionStatus`` value.
            user_id:    Optional owner identifier for ownership validation.

        Returns:
            Updated ``Session`` Pydantic model.

        Raises:
            SessionNotFoundError:  If the session does not exist.
            SessionOwnershipError: If *user_id* is provided and does not match.

        Requirements: 1.6, 1.7
        """
        db_session = self._fetch_session_or_raise(db, session_id)

        if user_id is not None and db_session.user_id != user_id:
            logger.warning(
                "Ownership violation on status update: user '%s' tried to update "
                "session %s (owned by '%s')",
                user_id,
                session_id,
                db_session.user_id,
            )
            raise SessionOwnershipError(
                f"Session '{session_id}' does not belong to user '{user_id}'."
            )

        db_session.status = status.value

        try:
            db.commit()
            db.refresh(db_session)
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error(
                "Failed to update status for session %s: %s", session_id, exc
            )
            raise

        logger.info("Session %s status updated to '%s'", session_id, status.value)
        return self._orm_to_pydantic(db_session)

    # ------------------------------------------------------------------
    # delete_session
    # ------------------------------------------------------------------

    def delete_session(
        self,
        db: DBSession,
        session_id: str,
        user_id: str,
    ) -> bool:
        """Delete a session and all its messages (cascade).

        Validates ownership before deletion.

        Args:
            db:         SQLAlchemy database session.
            session_id: UUID string of the session to delete.
            user_id:    Identifier of the authenticated user.

        Returns:
            ``True`` if the session was deleted successfully.

        Raises:
            SessionNotFoundError:  If the session does not exist.
            SessionOwnershipError: If the session belongs to a different user.

        Requirements: 1.8, 12.1, 12.2
        """
        db_session = self._fetch_session_or_raise(db, session_id)

        if db_session.user_id != user_id:
            logger.warning(
                "Ownership violation on delete: user '%s' tried to delete "
                "session %s (owned by '%s')",
                user_id,
                session_id,
                db_session.user_id,
            )
            raise SessionOwnershipError(
                f"Session '{session_id}' does not belong to user '{user_id}'."
            )

        try:
            db.delete(db_session)
            db.commit()
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error(
                "Failed to delete session %s: %s", session_id, exc
            )
            raise

        logger.info("Session %s deleted by user '%s'", session_id, user_id)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_session_or_raise(
        self,
        db: DBSession,
        session_id: str,
    ) -> object:
        """Fetch a session ORM object by ID or raise ``SessionNotFoundError``."""
        ORM = self._ORM

        # Validate UUID format
        try:
            session_uuid_str = str(uuid.UUID(str(session_id)))
        except ValueError:
            raise SessionNotFoundError(
                f"'{session_id}' is not a valid session UUID."
            )

        # Filter by string representation (works for both PostgreSQL and SQLite)
        db_session = (
            db.query(ORM)
            .filter(ORM.session_id == session_uuid_str)
            .first()
        )

        if db_session is None:
            raise SessionNotFoundError(
                f"Session '{session_id}' not found."
            )

        return db_session

    @staticmethod
    def _orm_to_pydantic(db_session) -> Session:
        """Convert a SQLAlchemy Session ORM object to a Pydantic ``Session``."""
        return Session(
            session_id=str(db_session.session_id),
            user_id=db_session.user_id,
            initial_query=db_session.initial_query,
            created_at=db_session.created_at,
            updated_at=db_session.updated_at,
            status=SessionStatus(db_session.status),
            message_count=db_session.message_count,
        )

    @staticmethod
    def _orm_to_metadata(db_session) -> SessionMetadata:
        """Convert a SQLAlchemy Session ORM object to a ``SessionMetadata``."""
        return SessionMetadata(
            session_id=str(db_session.session_id),
            initial_query=db_session.initial_query,
            created_at=db_session.created_at,
            updated_at=db_session.updated_at,
            status=SessionStatus(db_session.status),
            message_count=db_session.message_count,
        )
