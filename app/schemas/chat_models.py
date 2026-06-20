"""
Pydantic models for the Deep Research Memory Chatbot feature.

This module defines the data-transfer / validation models used across the
memory-chatbot layer.  They are intentionally separate from the SQLAlchemy
ORM models in ``app.core.session_models`` so that the API / service layer
never leaks database internals.

Models
------
SessionStatus   – enum for session lifecycle states
MessageRole     – enum for conversation turn roles
Session         – full session record returned by the API
SessionMetadata – lightweight session summary for list views
Message         – a single conversation turn
ContextWindow   – the in-memory context fed to agents
Source          – a cited source (arxiv paper or web URL)
ResearchReport  – the initial deep-research output stored per session

Requirements: 1.1, 2.1
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SessionStatus(str, Enum):
    """Lifecycle states for a conversation session.

    Values are lowercase strings so they round-trip cleanly through JSON and
    match the ``status`` column in the ``sessions`` table.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class MessageRole(str, Enum):
    """Who authored a particular message in the conversation.

    Values match the ``role`` column in the ``messages`` table.
    """

    USER = "user"
    ASSISTANT = "assistant"


# ---------------------------------------------------------------------------
# Source model (shared by Message and ResearchReport)
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """A single cited source referenced in a research report or message.

    Attributes:
        index:      1-based citation index used in the answer text (e.g. [3]).
        type:       ``"arxiv"`` or ``"url"``.
        title:      Human-readable title of the source.
        url:        Canonical URL for the source.
        arxiv_id:   arXiv identifier (only set when ``type == "arxiv"``).
        authors:    List of author names.
        year:       Publication year.
        venue:      Journal / conference name.
        abstract:   Short abstract or excerpt.
        citation_count: Number of citations (from Semantic Scholar / OpenAlex).
    """

    index: int = Field(..., ge=1, description="1-based citation index")
    type: str = Field(..., description="Source type: 'arxiv' or 'url'")
    title: str = Field(default="", description="Title of the source")
    url: str = Field(default="", description="Canonical URL")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv paper ID")
    authors: List[str] = Field(default_factory=list, description="Author names")
    year: Optional[int] = Field(default=None, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Journal or conference")
    abstract: Optional[str] = Field(default=None, description="Abstract or excerpt")
    citation_count: Optional[int] = Field(default=None, description="Citation count")


# ---------------------------------------------------------------------------
# Session models
# ---------------------------------------------------------------------------


class Session(BaseModel):
    """Full session record as returned by the API.

    Attributes:
        session_id:     UUID that uniquely identifies the session.
        user_id:        Identifier of the owning user.
        initial_query:  The research query that started the session.
        created_at:     When the session was created.
        updated_at:     When the session was last modified.
        status:         Current lifecycle state.
        message_count:  Total number of messages in the session.
    """

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID identifying the session",
    )
    user_id: str = Field(..., description="Owner user identifier")
    initial_query: str = Field(..., min_length=1, description="Original research query")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp (UTC)",
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last-updated timestamp (UTC)",
    )
    status: SessionStatus = Field(
        default=SessionStatus.ACTIVE,
        description="Session lifecycle state",
    )
    message_count: int = Field(
        default=0,
        ge=0,
        description="Number of messages in the conversation",
    )

    model_config = {"from_attributes": True}


class SessionMetadata(BaseModel):
    """Lightweight session summary used in list views.

    Contains only the fields needed to render a session list item in the UI.
    """

    session_id: str
    initial_query: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus
    message_count: int

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single turn in a conversation session.

    Attributes:
        message_id:   UUID that uniquely identifies the message.
        session_id:   UUID of the parent session.
        role:         Who authored the message (user or assistant).
        content:      Full text of the message.
        timestamp:    When the message was created.
        token_count:  Number of tokens in ``content`` (tracked for compression).
        metadata:     Arbitrary extra data – sources, confidence_score, flags, etc.
    """

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID identifying the message",
    )
    session_id: str = Field(..., description="UUID of the parent session")
    role: MessageRole = Field(..., description="Message author role")
    content: str = Field(..., min_length=1, description="Message text")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp (UTC)",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Token count for context-window management",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (sources, confidence_score, is_summary, …)",
    )

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# ResearchReport model
# ---------------------------------------------------------------------------


class ResearchReport(BaseModel):
    """The initial deep-research output stored at the start of a session.

    This object is kept intact in the context window and is *never* compressed
    or summarised (see context compression strategy in the design document).

    Attributes:
        answer:           The synthesised research answer.
        sources:          Ordered list of cited sources.
        planner_decision: Raw planner routing metadata.
        confidence_score: Quality score in [0, 1].
        review_feedback:  Reviewer commentary on the answer.
    """

    answer: str = Field(..., min_length=1, description="Synthesised research answer")
    sources: List[Source] = Field(
        default_factory=list,
        description="Cited sources in citation-index order",
    )
    planner_decision: Dict[str, Any] = Field(
        default_factory=dict,
        description="Planner routing metadata (strategy, reasoning, …)",
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Answer quality score between 0 and 1",
    )
    review_feedback: str = Field(
        default="",
        description="Reviewer commentary on the answer",
    )


# ---------------------------------------------------------------------------
# ContextWindow model
# ---------------------------------------------------------------------------


class ContextWindow(BaseModel):
    """The in-memory context fed to agents when processing a follow-up query.

    Holds the recent conversation history, the original research report, and
    aggregated source list.  The ``is_compressed`` flag indicates whether old
    messages have been replaced by a summary.

    Attributes:
        session_id:       UUID of the owning session.
        messages:         Recent conversation turns (up to ``max_messages``).
        research_report:  The initial research report (never compressed).
        sources:          All sources accumulated across the conversation.
        total_tokens:     Sum of ``token_count`` across all messages.
        is_compressed:    Whether older messages have been summarised.
    """

    session_id: str = Field(..., description="UUID of the owning session")
    messages: List[Message] = Field(
        default_factory=list,
        description="Recent conversation turns",
    )
    research_report: Optional[ResearchReport] = Field(
        default=None,
        description="Initial research report (preserved across compression)",
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="All sources accumulated in this session",
    )
    total_tokens: int = Field(
        default=0,
        ge=0,
        description="Total token count across all messages in the window",
    )
    is_compressed: bool = Field(
        default=False,
        description="True when older messages have been replaced by a summary",
    )
