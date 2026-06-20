"""
MemoryStore: Redis-based short-term memory for conversation sessions.

This module provides the MemoryStore class which manages short-term conversation
memory using Redis.  Each session's context is stored as a Redis Hash under the
key ``session:{session_id}:context`` and expires after 24 hours.

User-prefixed keys (``user:{user_id}:sessions``) are maintained as a Redis
Sorted Set to provide multi-user isolation and fast session listing.

Requirements: 2.1, 2.2, 2.3, 2.6, 12.5
"""

from __future__ import annotations

import json
import logging
import time
from typing import List, Optional

import redis

from app.schemas.chat_models import (
    ContextWindow,
    Message,
    MessageRole,
    ResearchReport,
    Source,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS: int = 86_400  # 24 hours
USER_SESSIONS_TTL_SECONDS: int = 7 * 86_400  # 7 days
DEFAULT_MAX_MESSAGES: int = 10

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MemoryStoreError(Exception):
    """Base exception for MemoryStore errors."""


class SessionContextNotFoundError(MemoryStoreError):
    """Raised when no context exists for the given session_id."""


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _context_key(session_id: str) -> str:
    """Redis Hash key for a session's context.

    Key structure: ``session:{session_id}:context``

    Requirements: 2.6 (Redis key structure)
    """
    return f"session:{session_id}:context"


def _user_sessions_key(user_id: str) -> str:
    """Redis Sorted Set key for a user's session list.

    Key structure: ``user:{user_id}:sessions``

    Requirements: 12.5 (user-prefixed keys for multi-user isolation)
    """
    return f"user:{user_id}:sessions"


def _source_key(session_id: str, source_index: int) -> str:
    """Redis String key for a cached source.

    Key structure: ``session:{session_id}:source:{source_index}``
    """
    return f"session:{session_id}:source:{source_index}"


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------


class MemoryStore:
    """Redis-backed short-term memory store for conversation sessions.

    Each session's context is stored as a Redis Hash with the following fields:

    - ``messages``        – JSON-serialised list of ``Message`` objects
    - ``research_report`` – JSON-serialised ``ResearchReport`` (or empty string)
    - ``sources``         – JSON-serialised list of ``Source`` objects
    - ``total_tokens``    – integer stored as a string
    - ``last_updated``    – Unix timestamp (float) stored as a string
    - ``is_compressed``   – ``"1"`` or ``"0"``

    All keys are set with a 24-hour TTL (refreshed on every write).

    User isolation is enforced by maintaining a per-user Sorted Set
    (``user:{user_id}:sessions``) whose members are session IDs and whose
    scores are creation timestamps.  This set is *separate* from the session
    context hash and is used only for listing sessions; it does **not** gate
    access to context data (that is the responsibility of SessionManager).

    Args:
        redis_client: A ``redis.Redis`` instance.  Callers are responsible for
            creating and closing the connection.

    Requirements: 2.1, 2.2, 2.3, 2.6, 12.5
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    # ------------------------------------------------------------------
    # save_message
    # ------------------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        message: Message,
        user_id: Optional[str] = None,
    ) -> None:
        """Append *message* to the session's context in Redis.

        If no context exists for *session_id* yet, a new Hash is created.
        The TTL is refreshed to 24 hours on every call.

        When *user_id* is provided the session is also registered in the
        user's session Sorted Set (``user:{user_id}:sessions``) so that
        ``list_user_sessions()`` can return it.

        Args:
            session_id: UUID string identifying the session.
            message:    ``Message`` Pydantic model to persist.
            user_id:    Optional owner identifier for multi-user isolation.

        Requirements: 2.1, 2.6, 12.5
        """
        key = _context_key(session_id)

        # Load existing messages (or start fresh)
        raw = self._redis.hget(key, "messages")
        messages: List[dict] = json.loads(raw) if raw else []

        # Append the new message
        messages.append(message.model_dump(mode="json"))

        # Recalculate total_tokens
        total_tokens = sum(m.get("token_count", 0) for m in messages)

        # Persist back to Redis as a Hash
        self._redis.hset(
            key,
            mapping={
                "messages": json.dumps(messages),
                "total_tokens": str(total_tokens),
                "last_updated": str(time.time()),
            },
        )

        # Refresh TTL
        self._redis.expire(key, SESSION_TTL_SECONDS)

        # Register session in user's sorted set (for multi-user isolation)
        if user_id is not None:
            user_key = _user_sessions_key(user_id)
            # Score = current timestamp so sessions are ordered by activity
            self._redis.zadd(user_key, {session_id: time.time()})
            self._redis.expire(user_key, USER_SESSIONS_TTL_SECONDS)

        logger.debug(
            "Saved message %s to session %s (total messages: %d)",
            message.message_id,
            session_id,
            len(messages),
        )

    # ------------------------------------------------------------------
    # get_context_window
    # ------------------------------------------------------------------

    def get_context_window(
        self,
        session_id: str,
        max_messages: int = DEFAULT_MAX_MESSAGES,
    ) -> ContextWindow:
        """Retrieve the context window for *session_id*.

        Returns a ``ContextWindow`` containing the last *max_messages*
        messages, the stored ``ResearchReport`` (if any), accumulated
        sources, total token count, and compression flag.

        Args:
            session_id:   UUID string identifying the session.
            max_messages: Maximum number of recent messages to include
                          (default 10, per Requirement 2.2).

        Returns:
            A ``ContextWindow`` Pydantic model.

        Raises:
            SessionContextNotFoundError: If no context exists for the session.

        Requirements: 2.2, 2.3
        """
        key = _context_key(session_id)

        # Fetch all fields in one round-trip
        data = self._redis.hgetall(key)

        if not data:
            raise SessionContextNotFoundError(
                f"No context found in Redis for session '{session_id}'."
            )

        # Decode bytes → str (redis-py returns bytes by default)
        data = {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in data.items()
        }

        # Deserialise messages
        raw_messages: List[dict] = json.loads(data.get("messages", "[]"))
        all_messages = [Message(**m) for m in raw_messages]

        # Apply sliding window — return only the last N messages
        windowed_messages = all_messages[-max_messages:] if max_messages > 0 else all_messages

        # Deserialise research_report (may be absent)
        research_report: Optional[ResearchReport] = None
        raw_report = data.get("research_report", "")
        if raw_report:
            research_report = ResearchReport(**json.loads(raw_report))

        # Deserialise sources
        raw_sources: List[dict] = json.loads(data.get("sources", "[]"))
        sources = [Source(**s) for s in raw_sources]

        # Parse scalar fields
        total_tokens = int(data.get("total_tokens", "0"))
        is_compressed = data.get("is_compressed", "0") == "1"

        return ContextWindow(
            session_id=session_id,
            messages=windowed_messages,
            research_report=research_report,
            sources=sources,
            total_tokens=total_tokens,
            is_compressed=is_compressed,
        )

    # ------------------------------------------------------------------
    # init_session_context
    # ------------------------------------------------------------------

    def init_session_context(
        self,
        session_id: str,
        research_report: Optional[ResearchReport] = None,
        sources: Optional[List[Source]] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Initialise a fresh context Hash for a new session.

        Should be called once when a session is created (after the initial
        research report is generated).  Idempotent: calling it again on an
        existing session overwrites the stored data.

        Args:
            session_id:      UUID string identifying the session.
            research_report: Optional initial ``ResearchReport`` to store.
            sources:         Optional initial list of ``Source`` objects.
            user_id:         Optional owner identifier for multi-user isolation.

        Requirements: 1.3, 2.3, 2.6
        """
        key = _context_key(session_id)

        mapping: dict = {
            "messages": json.dumps([]),
            "total_tokens": "0",
            "last_updated": str(time.time()),
            "is_compressed": "0",
            "sources": json.dumps([s.model_dump(mode="json") for s in (sources or [])]),
        }

        if research_report is not None:
            mapping["research_report"] = research_report.model_dump_json()
        else:
            mapping["research_report"] = ""

        self._redis.hset(key, mapping=mapping)
        self._redis.expire(key, SESSION_TTL_SECONDS)

        # Register in user's sorted set
        if user_id is not None:
            user_key = _user_sessions_key(user_id)
            self._redis.zadd(user_key, {session_id: time.time()})
            self._redis.expire(user_key, USER_SESSIONS_TTL_SECONDS)

        logger.info("Initialised context for session %s", session_id)

    # ------------------------------------------------------------------
    # save_research_report
    # ------------------------------------------------------------------

    def save_research_report(
        self,
        session_id: str,
        research_report: ResearchReport,
    ) -> None:
        """Persist the initial research report for a session.

        The report is stored in the ``research_report`` field of the session
        Hash and is *never* compressed (per design decision).

        Args:
            session_id:      UUID string identifying the session.
            research_report: ``ResearchReport`` to store.

        Requirements: 2.3
        """
        key = _context_key(session_id)
        self._redis.hset(key, "research_report", research_report.model_dump_json())
        self._redis.expire(key, SESSION_TTL_SECONDS)
        logger.debug("Saved research report for session %s", session_id)

    # ------------------------------------------------------------------
    # update_sources
    # ------------------------------------------------------------------

    def update_sources(
        self,
        session_id: str,
        sources: List[Source],
    ) -> None:
        """Replace the sources list for a session.

        Args:
            session_id: UUID string identifying the session.
            sources:    Updated list of ``Source`` objects.

        Requirements: 2.7
        """
        key = _context_key(session_id)
        self._redis.hset(
            key,
            "sources",
            json.dumps([s.model_dump(mode="json") for s in sources]),
        )
        self._redis.expire(key, SESSION_TTL_SECONDS)

    # ------------------------------------------------------------------
    # delete_session_context
    # ------------------------------------------------------------------

    def delete_session_context(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Remove all Redis data for a session.

        Deletes the context Hash and removes the session from the user's
        Sorted Set (if *user_id* is provided).

        Args:
            session_id: UUID string identifying the session.
            user_id:    Optional owner identifier.
        """
        key = _context_key(session_id)
        self._redis.delete(key)

        if user_id is not None:
            user_key = _user_sessions_key(user_id)
            self._redis.zrem(user_key, session_id)

        logger.info("Deleted Redis context for session %s", session_id)

    # ------------------------------------------------------------------
    # list_user_sessions
    # ------------------------------------------------------------------

    def list_user_sessions(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[str]:
        """Return session IDs for *user_id*, ordered by most-recent activity.

        Uses the ``user:{user_id}:sessions`` Sorted Set.  Sessions are scored
        by their last-activity timestamp so the most recently active sessions
        appear first.

        Args:
            user_id: Owner identifier.
            limit:   Maximum number of session IDs to return.

        Returns:
            List of session ID strings, newest first.

        Requirements: 12.5
        """
        user_key = _user_sessions_key(user_id)
        # ZREVRANGE returns members ordered by score descending (newest first)
        raw = self._redis.zrevrange(user_key, 0, limit - 1)
        return [
            (r.decode() if isinstance(r, bytes) else r) for r in raw
        ]

    # ------------------------------------------------------------------
    # save_source
    # ------------------------------------------------------------------

    def save_source(
        self,
        session_id: str,
        source: Source,
    ) -> int:
        """Cache a source in Redis and return its (possibly existing) index.

        Implements source deduplication: if an identical source (matched by
        ``arxiv_id`` for arXiv papers or ``url`` for web sources) is already
        cached for this session, the existing index is returned and no new key
        is written.

        The source is stored as a JSON string under the key
        ``session:{session_id}:source:{source_index}`` with a 24-hour TTL.

        The session context Hash is also updated so that the ``sources`` field
        stays in sync with the individual source keys.

        Args:
            session_id: UUID string identifying the session.
            source:     ``Source`` Pydantic model to cache.

        Returns:
            The 1-based citation index for the source (existing or newly
            assigned).

        Requirements: 2.7, 10.6
        """
        # --- Deduplication: scan existing sources in the context Hash -------
        existing_sources = self._get_sources_from_context(session_id)
        for existing in existing_sources:
            if self._is_same_source(source, existing):
                logger.debug(
                    "Source deduplication: session %s already has source at index %d",
                    session_id,
                    existing.index,
                )
                return existing.index

        # --- Assign a new index (next available) ----------------------------
        # The new index is one beyond the current maximum, or 1 if no sources.
        new_index = (max((s.index for s in existing_sources), default=0) + 1)

        # Ensure the source object carries the correct index
        source_to_store = source.model_copy(update={"index": new_index})

        # --- Write individual source key ------------------------------------
        source_key = _source_key(session_id, new_index)
        self._redis.set(
            source_key,
            source_to_store.model_dump_json(),
            ex=SESSION_TTL_SECONDS,
        )

        # --- Update the sources list in the context Hash --------------------
        updated_sources = existing_sources + [source_to_store]
        context_key = _context_key(session_id)
        self._redis.hset(
            context_key,
            "sources",
            json.dumps([s.model_dump(mode="json") for s in updated_sources]),
        )
        self._redis.expire(context_key, SESSION_TTL_SECONDS)

        logger.debug(
            "Cached source '%s' at index %d for session %s",
            source_to_store.title or source_to_store.url,
            new_index,
            session_id,
        )
        return new_index

    # ------------------------------------------------------------------
    # get_source
    # ------------------------------------------------------------------

    def get_source(
        self,
        session_id: str,
        source_index: int,
    ) -> Optional[Source]:
        """Retrieve a cached source by its citation index.

        Looks up the individual source key
        ``session:{session_id}:source:{source_index}``.  Falls back to
        scanning the context Hash ``sources`` field if the individual key has
        expired or was never written.

        Args:
            session_id:   UUID string identifying the session.
            source_index: 1-based citation index of the source.

        Returns:
            The ``Source`` Pydantic model, or ``None`` if not found.

        Requirements: 2.7, 10.6
        """
        # Primary lookup: individual source key
        source_key = _source_key(session_id, source_index)
        raw = self._redis.get(source_key)
        if raw:
            data = raw.decode() if isinstance(raw, bytes) else raw
            return Source(**json.loads(data))

        # Fallback: scan the sources list in the context Hash
        for source in self._get_sources_from_context(session_id):
            if source.index == source_index:
                logger.debug(
                    "get_source: found source %d via context Hash fallback for session %s",
                    source_index,
                    session_id,
                )
                return source

        logger.debug(
            "get_source: source index %d not found for session %s",
            source_index,
            session_id,
        )
        return None

    # ------------------------------------------------------------------
    # get_all_sources
    # ------------------------------------------------------------------

    def get_all_sources(self, session_id: str) -> List[Source]:
        """Return all cached sources for a session, ordered by index.

        Reads the ``sources`` field from the session context Hash.

        Args:
            session_id: UUID string identifying the session.

        Returns:
            List of ``Source`` objects sorted by ``index`` ascending.
            Returns an empty list if the session has no sources or does not
            exist.

        Requirements: 2.7, 10.6
        """
        sources = self._get_sources_from_context(session_id)
        return sorted(sources, key=lambda s: s.index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_sources_from_context(self, session_id: str) -> List[Source]:
        """Read the ``sources`` field from the context Hash.

        Returns an empty list if the session does not exist or has no sources.
        """
        key = _context_key(session_id)
        raw = self._redis.hget(key, "sources")
        if not raw:
            return []
        data = raw.decode() if isinstance(raw, bytes) else raw
        raw_sources: List[dict] = json.loads(data)
        return [Source(**s) for s in raw_sources]

    @staticmethod
    def _is_same_source(s1: Source, s2: Source) -> bool:
        """Return ``True`` when two sources refer to the same content.

        Comparison rules (matching the SourceManager design):
        - arXiv papers: compare ``arxiv_id`` (case-insensitive).
        - Web URLs: compare ``url`` (exact match after stripping trailing slash).
        - Mixed types: always ``False``.

        Requirements: 10.6 (deduplication)
        """
        if s1.type == "arxiv" and s2.type == "arxiv":
            if s1.arxiv_id and s2.arxiv_id:
                return s1.arxiv_id.lower() == s2.arxiv_id.lower()
            # Fall back to URL comparison if arxiv_id is missing
            return s1.url.rstrip("/") == s2.url.rstrip("/")
        if s1.type == "url" and s2.type == "url":
            return s1.url.rstrip("/") == s2.url.rstrip("/")
        return False

    # ------------------------------------------------------------------
    # session_exists
    # ------------------------------------------------------------------

    def session_exists(self, session_id: str) -> bool:
        """Return ``True`` if a context Hash exists for *session_id*.

        Args:
            session_id: UUID string identifying the session.
        """
        return bool(self._redis.exists(_context_key(session_id)))

    # ------------------------------------------------------------------
    # refresh_ttl
    # ------------------------------------------------------------------

    def refresh_ttl(self, session_id: str) -> None:
        """Reset the TTL on a session's context key to 24 hours.

        Useful when a session is accessed without writing new messages.

        Args:
            session_id: UUID string identifying the session.
        """
        key = _context_key(session_id)
        self._redis.expire(key, SESSION_TTL_SECONDS)
