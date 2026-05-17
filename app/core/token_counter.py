"""
Token counting utilities for context window management.

Uses tiktoken (cl100k_base encoding) to count tokens accurately.
Falls back to a simple word-based estimate if tiktoken fails.

Requirements: 2.4, 11.1
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.chat_models import ContextWindow, Message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encoding initialisation (module-level, loaded once)
# ---------------------------------------------------------------------------

_encoding = None


def _get_encoding():
    """Return the cl100k_base tiktoken encoding, loading it on first call."""
    global _encoding
    if _encoding is None:
        try:
            import tiktoken
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # pragma: no cover
            logger.warning("tiktoken unavailable, falling back to word estimate: %s", exc)
    return _encoding


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in *text*.

    Uses tiktoken with the ``cl100k_base`` encoding (used by GPT-3.5 and
    GPT-4).  If tiktoken is unavailable or raises an error, falls back to
    ``len(text.split()) * 2`` as a rough estimate.

    Args:
        text:  The string to tokenise.
        model: Unused – kept for API compatibility.  The encoding is always
               ``cl100k_base`` regardless of model name.

    Returns:
        Integer token count (>= 0).
    """
    if not text:
        return 0

    enc = _get_encoding()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception as exc:
            logger.warning("tiktoken encode failed, using fallback: %s", exc)

    # Fallback: word count * 2
    return len(text.split()) * 2


def count_message_tokens(message: "Message") -> int:
    """Count tokens in *message* and store the result in its metadata.

    The token count is written to ``message.metadata["token_count"]`` so that
    callers can persist it alongside the message without a second pass.

    Args:
        message: A ``Message`` Pydantic model.  ``message.metadata`` is
                 mutated in-place.

    Returns:
        Integer token count for ``message.content``.
    """
    token_count = count_tokens(message.content)
    message.metadata["token_count"] = token_count
    return token_count


def calculate_total_tokens(context_window: "ContextWindow") -> int:
    """Sum token counts across all messages and the research report.

    For each message, ``count_message_tokens`` is called so that
    ``message.metadata["token_count"]`` is populated as a side-effect.

    The research report's ``answer`` field is also counted when present.

    Args:
        context_window: A ``ContextWindow`` Pydantic model.

    Returns:
        Total integer token count.
    """
    total = 0

    for message in context_window.messages:
        total += count_message_tokens(message)

    if context_window.research_report is not None:
        total += count_tokens(context_window.research_report.answer)

    return total
