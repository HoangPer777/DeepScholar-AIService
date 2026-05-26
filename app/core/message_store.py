"""
MessageStore: PostgreSQL-backed message persistence for chat history.

This module provides the MessageStore class which handles saving and retrieving
conversation messages from the PostgreSQL database, enabling persistent chat
history that survives Docker restarts.

The dual-write pattern is:
  POST /api/chat/ → save to Redis (TTL 24h) + save to PostgreSQL (permanent)
  GET /api/chat/history/{session_id} → try Redis first → fallback to PostgreSQL

Requirements: 2.2, 2.3
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from sqlalchemy.orm import Session as DBSession
from sqlalchemy.exc import SQLAlchemyError

from app.core.session_models import Session as SessionORM, Message as MessageORM
from app.core.session_manager import SessionManager, SessionNotFoundError
from app.schemas.chat_models import Message, MessageRole

logger = logging.getLogger(__name__)


class MessageStore:
    """PostgreSQL-backed message persistence for chat history.

    Handles saving and retrieving conversation messages from PostgreSQL,
    providing durable storage that persists across Docker restarts.

    Args:
        session_manager: Optional SessionManager instance. If not provided,
            a default instance is created.
    """

    def __init__(self, session_manager: Optional[SessionManager] = None) -> None:
        self._session_manager = session_manager or SessionManager()

    def save_message(
        self,
        db: DBSession,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> MessageORM:
        """Save a message to the PostgreSQL messages table.

        Args:
            db:         SQLAlchemy database session.
            session_id: UUID string of the parent session.
            role:       Message role: 'user' or 'assistant'.
            content:    Full text content of the message.
            metadata:   Optional arbitrary JSON metadata.

        Returns:
            The persisted MessageORM object.

        Raises:
            SQLAlchemyError: On unexpected database errors.
        """
        message_id = str(uuid.uuid4())
        db_message = MessageORM(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            token_count=0,
            msg_metadata=metadata if metadata else None,
        )

        try:
            db.add(db_message)
            db.commit()
            db.refresh(db_message)
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error(
                "Failed to save message for session '%s': %s", session_id, exc
            )
            raise

        logger.debug(
            "Saved %s message %s for session %s",
            role,
            message_id,
            session_id,
        )
        return db_message

    def get_messages(
        self,
        db: DBSession,
        session_id: str,
        limit: int = 100,
    ) -> List[MessageORM]:
        """Get all messages for a session from PostgreSQL, ordered by timestamp.

        Args:
            db:         SQLAlchemy database session.
            session_id: UUID string of the session.
            limit:      Maximum number of messages to return (default 100).

        Returns:
            List of MessageORM objects ordered by timestamp ascending.
        """
        try:
            messages = (
                db.query(MessageORM)
                .filter(MessageORM.session_id == session_id)
                .order_by(MessageORM.timestamp.asc())
                .limit(limit)
                .all()
            )
            return messages
        except SQLAlchemyError as exc:
            logger.error(
                "Failed to get messages for session '%s': %s", session_id, exc
            )
            raise

    def ensure_session_exists(
        self,
        db: DBSession,
        session_id: str,
        user_id: str,
        initial_query: str,
    ) -> SessionORM:
        """Create session in PostgreSQL if it doesn't exist.

        Checks if a session with the given session_id exists. If not, creates
        a new session record. If it does exist, returns the existing one.

        Args:
            db:            SQLAlchemy database session.
            session_id:    UUID string of the session to ensure exists.
            user_id:       Owner user identifier.
            initial_query: The research query that started this session.

        Returns:
            The existing or newly created SessionORM object.

        Raises:
            SQLAlchemyError: On unexpected database errors.
        """
        # Check if session already exists
        existing = (
            db.query(SessionORM)
            .filter(SessionORM.session_id == session_id)
            .first()
        )

        if existing is not None:
            return existing

        # Create new session
        db_session = SessionORM(
            session_id=session_id,
            user_id=user_id,
            initial_query=initial_query,
            status="active",
            message_count=0,
        )

        try:
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error(
                "Failed to create session '%s' for user '%s': %s",
                session_id,
                user_id,
                exc,
            )
            raise

        logger.info(
            "Created session %s for user '%s' via ensure_session_exists",
            session_id,
            user_id,
        )
        return db_session

    def messages_to_pydantic(self, db_messages: List[MessageORM]) -> List[Message]:
        """Convert a list of MessageORM objects to Pydantic Message models.

        Args:
            db_messages: List of SQLAlchemy MessageORM objects.

        Returns:
            List of Pydantic Message models.
        """
        result = []
        for m in db_messages:
            result.append(
                Message(
                    message_id=str(m.message_id),
                    session_id=str(m.session_id),
                    role=MessageRole(m.role),
                    content=m.content,
                    timestamp=m.timestamp,
                    token_count=m.token_count or 0,
                    metadata=m.msg_metadata or {},
                )
            )
        return result
