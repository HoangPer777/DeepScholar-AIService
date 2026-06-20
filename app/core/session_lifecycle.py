"""
Session lifecycle management for the Deep Research Memory Chatbot.

This module provides:

- ``mark_inactive_sessions(db)``
    Queries all sessions where ``updated_at < now - 24h`` AND
    ``status == 'active'`` and marks them as ``'inactive'``.
    Intended to be called periodically by a background scheduler.

- ``increment_message_count(db, session_id)``
    Atomically increments ``message_count`` and refreshes ``updated_at``
    for the given session.  Called whenever a new message is appended to
    a session so that the session metadata stays accurate.

- ``start_lifecycle_scheduler(app)``
    Registers a FastAPI ``lifespan``-compatible background task (using
    ``asyncio`` + ``asyncio.create_task``) that runs
    ``mark_inactive_sessions`` every hour.  The scheduler is started when
    the FastAPI application starts and cancelled on shutdown.

Requirements: 1.7, 1.8
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session as DBSession
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INACTIVITY_THRESHOLD_HOURS: int = 24
"""Sessions inactive for longer than this many hours are marked 'inactive'."""

SCHEDULER_INTERVAL_SECONDS: int = 3600
"""How often (in seconds) the background scheduler runs the cleanup job."""


# ---------------------------------------------------------------------------
# mark_inactive_sessions
# ---------------------------------------------------------------------------


def mark_inactive_sessions(db: DBSession, orm_class=None) -> int:
    """Mark active sessions that have been idle for ≥ 24 hours as inactive.

    Queries the ``sessions`` table for rows where:
    - ``status == 'active'``
    - ``updated_at < now(UTC) - 24 hours``

    and bulk-updates their ``status`` to ``'inactive'``.

    Args:
        db:        SQLAlchemy database session (caller controls the transaction).
        orm_class: Optional ORM class override (used in tests to inject a
                   SQLite-compatible model instead of the production one).

    Returns:
        The number of sessions that were marked inactive.

    Raises:
        SQLAlchemyError: On unexpected database errors (caller should handle).

    Requirements: 1.7
    """
    if orm_class is None:
        # Import lazily to avoid circular imports at module load time
        from app.core.session_models import Session as _SessionORM
        orm_class = _SessionORM

    cutoff = datetime.now(timezone.utc) - timedelta(hours=INACTIVITY_THRESHOLD_HOURS)

    try:
        # Fetch matching rows so we can log them and update individually
        # (bulk UPDATE via query.update() is used for efficiency)
        rows_updated = (
            db.query(orm_class)
            .filter(
                orm_class.status == "active",
                orm_class.updated_at < cutoff,
            )
            .update({"status": "inactive"}, synchronize_session="fetch")
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.error("mark_inactive_sessions failed: %s", exc)
        raise

    if rows_updated:
        logger.info(
            "mark_inactive_sessions: marked %d session(s) as inactive "
            "(cutoff=%s UTC)",
            rows_updated,
            cutoff.isoformat(),
        )
    return rows_updated


# ---------------------------------------------------------------------------
# increment_message_count
# ---------------------------------------------------------------------------


def increment_message_count(
    db: DBSession,
    session_id: str,
    orm_class=None,
) -> bool:
    """Increment ``message_count`` and refresh ``updated_at`` for a session.

    This should be called every time a new message is appended to a session
    so that the session metadata (visible in the session list) stays accurate.

    Args:
        db:         SQLAlchemy database session.
        session_id: UUID string of the session to update.
        orm_class:  Optional ORM class override for testing.

    Returns:
        ``True`` if the session was found and updated, ``False`` if the
        session does not exist.

    Raises:
        SQLAlchemyError: On unexpected database errors.

    Requirements: 1.8
    """
    if orm_class is None:
        from app.core.session_models import Session as _SessionORM
        orm_class = _SessionORM

    try:
        session_row = (
            db.query(orm_class)
            .filter(orm_class.session_id == session_id)
            .first()
        )

        if session_row is None:
            logger.warning(
                "increment_message_count: session '%s' not found", session_id
            )
            return False

        session_row.message_count = (session_row.message_count or 0) + 1
        session_row.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(session_row)

    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(
            "increment_message_count failed for session '%s': %s",
            session_id,
            exc,
        )
        raise

    logger.debug(
        "Session '%s' message_count incremented to %d",
        session_id,
        session_row.message_count,
    )
    return True


# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------


async def _lifecycle_loop(get_db_func, orm_class=None) -> None:
    """Async loop that periodically calls ``mark_inactive_sessions``.

    Args:
        get_db_func: A zero-argument callable that returns a SQLAlchemy
                     ``Session`` (database session).  The session is closed
                     after each run.
        orm_class:   Optional ORM class override (for testing).
    """
    logger.info(
        "Session lifecycle scheduler started (interval=%ds)",
        SCHEDULER_INTERVAL_SECONDS,
    )
    while True:
        await asyncio.sleep(SCHEDULER_INTERVAL_SECONDS)
        db = get_db_func()
        try:
            count = mark_inactive_sessions(db, orm_class=orm_class)
            logger.debug("Lifecycle scheduler: %d session(s) marked inactive", count)
        except Exception as exc:  # noqa: BLE001
            logger.error("Lifecycle scheduler error: %s", exc)
        finally:
            db.close()


def start_lifecycle_scheduler(get_db_func, orm_class=None) -> asyncio.Task:
    """Start the background lifecycle scheduler as an asyncio Task.

    This function is designed to be called from a FastAPI ``lifespan``
    context manager (or ``startup`` event handler) so that the scheduler
    runs for the lifetime of the application.

    Example usage in a FastAPI app::

        from contextlib import asynccontextmanager
        from fastapi import FastAPI
        from app.core.session_lifecycle import start_lifecycle_scheduler
        from app.core.database import SessionLocal

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            task = start_lifecycle_scheduler(SessionLocal)
            yield
            task.cancel()

        app = FastAPI(lifespan=lifespan)

    Args:
        get_db_func: A zero-argument callable that returns a SQLAlchemy
                     ``Session``.  Typically ``SessionLocal`` from
                     ``app.core.database``.
        orm_class:   Optional ORM class override (for testing).

    Returns:
        The running ``asyncio.Task`` so the caller can cancel it on shutdown.
    """
    task = asyncio.create_task(
        _lifecycle_loop(get_db_func, orm_class=orm_class),
        name="session_lifecycle_scheduler",
    )
    return task
