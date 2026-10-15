"""Supabase PostgreSQL checkpoint lifecycle for the Deep Research graph."""

from contextlib import ExitStack
from typing import Optional

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import settings

_stack: Optional[ExitStack] = None
_saver = None


def get_graph_checkpointer():
    """Return the singleton saver and initialize its Supabase tables once."""
    global _stack, _saver
    if _saver is None:
        # Unit tests may replace DATABASE_URL with SQLite. Production uses the
        # configured Supabase PostgreSQL URL and never takes this branch.
        if not settings.DATABASE_URL.startswith(("postgresql://", "postgres://")):
            _saver = InMemorySaver()
            return _saver
        _stack = ExitStack()
        _saver = _stack.enter_context(PostgresSaver.from_conn_string(settings.DATABASE_URL))
        _saver.setup()
    return _saver


def close_graph_checkpointer() -> None:
    """Close the pooled saver connection during FastAPI shutdown."""
    global _stack, _saver
    if _stack is not None:
        _stack.close()
    _stack = None
    _saver = None
