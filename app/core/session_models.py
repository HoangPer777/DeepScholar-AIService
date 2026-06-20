"""
Database models for conversation session management.

This module defines the SQLAlchemy models for the Deep Research Memory Chatbot feature,
including sessions table for managing conversation sessions, and messages table for
storing individual conversation messages.
"""

from sqlalchemy import Column, String, Text, DateTime, Integer, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.core.database import Base


class Session(Base):
    """
    Conversation session model for Deep Research Memory Chatbot.
    
    Each session represents a research conversation initiated by a user.
    Sessions track the initial query, status, and message count.
    
    Attributes:
        session_id: Unique identifier for the session (UUID)
        user_id: Identifier of the user who owns this session
        initial_query: The original research query that started this session
        created_at: Timestamp when the session was created
        updated_at: Timestamp when the session was last updated
        status: Current status of the session (active, inactive, archived)
        message_count: Number of messages in this conversation
    """
    __tablename__ = "sessions"

    session_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier for the conversation session"
    )
    
    user_id = Column(
        String(255),
        nullable=False,
        index=True,
        comment="User identifier from authentication system"
    )
    
    initial_query = Column(
        Text,
        nullable=False,
        comment="The original research query that initiated this session"
    )
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the session was created"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Timestamp when the session was last updated"
    )
    
    status = Column(
        String(20),
        nullable=False,
        default='active',
        comment="Session status: active, inactive, or archived"
    )
    
    message_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of messages in this conversation"
    )

    __table_args__ = (
        # Index for querying user's sessions ordered by creation time
        Index('idx_user_sessions', 'user_id', 'created_at'),
        
        # Index for querying sessions by status and update time
        # Useful for finding inactive sessions to archive
        Index('idx_session_status', 'status', 'updated_at'),
    )

    # Relationship to messages (one session has many messages)
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def __repr__(self):
        return f"<Session(session_id={self.session_id}, user_id={self.user_id}, status={self.status})>"


class Message(Base):
    """
    Conversation message model for Deep Research Memory Chatbot.

    Each message represents a single turn in a conversation session.
    Messages are linked to their parent session via a foreign key with
    cascade delete, so removing a session removes all its messages.

    Attributes:
        message_id: Unique identifier for the message (UUID)
        session_id: Foreign key referencing the parent session
        role: Who sent the message – 'user' or 'assistant'
        content: Full text content of the message
        timestamp: When the message was created
        token_count: Number of tokens in the message content
        metadata: Arbitrary JSON payload (sources, confidence_score, etc.)
    """

    __tablename__ = "messages"

    message_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Unique identifier for the message",
    )

    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key to the parent conversation session",
    )

    role = Column(
        String(20),
        nullable=False,
        comment="Message role: 'user' or 'assistant'",
    )

    content = Column(
        Text,
        nullable=False,
        comment="Full text content of the message",
    )

    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the message was created",
    )

    token_count = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of tokens in the message content",
    )

    msg_metadata = Column(
        "metadata",  # actual DB column name stays 'metadata'
        JSONB,
        nullable=True,
        comment="Arbitrary JSON metadata (sources, confidence_score, is_summary, etc.)",
    )

    # Relationship back to the parent session
    session = relationship("Session", back_populates="messages")

    __table_args__ = (
        # Index for retrieving all messages in a session ordered by time
        Index("idx_session_messages", "session_id", "timestamp"),
        # Index for time-based queries across all messages
        Index("idx_message_timestamp", "timestamp"),
    )

    def __repr__(self):
        return (
            f"<Message(message_id={self.message_id}, "
            f"session_id={self.session_id}, role={self.role})>"
        )
