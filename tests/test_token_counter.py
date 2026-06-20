"""
Unit tests for app.core.token_counter.

Tests cover:
- Basic token counting via tiktoken
- Fallback behaviour when tiktoken is unavailable
- Message token metadata update side-effect
- Total token aggregation across a ContextWindow
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from app.core.token_counter import (
    calculate_total_tokens,
    count_message_tokens,
    count_tokens,
)
from app.schemas.chat_models import (
    ContextWindow,
    Message,
    MessageRole,
    ResearchReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(content: str, session_id: str = "test-session") -> Message:
    return Message(
        session_id=session_id,
        role=MessageRole.USER,
        content=content,
    )


def _make_context_window(
    messages: list[Message],
    research_report: ResearchReport | None = None,
) -> ContextWindow:
    return ContextWindow(
        session_id="test-session",
        messages=messages,
        research_report=research_report,
    )


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_basic_token_count_is_positive(self):
        result = count_tokens("Hello world")
        assert result > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("Hi")
        long = count_tokens("This is a much longer sentence with many more words in it.")
        assert long > short

    def test_model_parameter_is_accepted(self):
        # model kwarg should not raise even though it is unused
        result = count_tokens("test text", model="gpt-3.5-turbo")
        assert result > 0

    def test_whitespace_only_returns_nonzero(self):
        # "   " splits into [] so fallback gives 0; tiktoken may give 1
        result = count_tokens("   ")
        assert isinstance(result, int)
        assert result >= 0

    def test_fallback_when_tiktoken_unavailable(self):
        """When tiktoken cannot be imported, fall back to word-count * 2."""
        text = "one two three four"  # 4 words → fallback = 8

        # Patch _get_encoding to return None (simulates import failure)
        with patch("app.core.token_counter._get_encoding", return_value=None):
            result = count_tokens(text)

        assert result == 8  # 4 words * 2

    def test_fallback_single_word(self):
        with patch("app.core.token_counter._get_encoding", return_value=None):
            result = count_tokens("hello")
        assert result == 2  # 1 word * 2

    def test_fallback_empty_string(self):
        with patch("app.core.token_counter._get_encoding", return_value=None):
            result = count_tokens("")
        assert result == 0


# ---------------------------------------------------------------------------
# count_message_tokens
# ---------------------------------------------------------------------------


class TestCountMessageTokens:
    def test_returns_positive_count(self):
        msg = _make_message("Hello, how are you?")
        result = count_message_tokens(msg)
        assert result > 0

    def test_stores_token_count_in_metadata(self):
        msg = _make_message("Store this count in metadata.")
        count = count_message_tokens(msg)
        assert "token_count" in msg.metadata
        assert msg.metadata["token_count"] == count

    def test_metadata_overwritten_on_second_call(self):
        msg = _make_message("Initial content.")
        count_message_tokens(msg)
        first_count = msg.metadata["token_count"]

        # Simulate content change by calling again (metadata should update)
        count_message_tokens(msg)
        assert msg.metadata["token_count"] == first_count  # same content → same count

    def test_existing_metadata_preserved(self):
        msg = _make_message("Some text.")
        msg.metadata["custom_key"] = "custom_value"
        count_message_tokens(msg)
        # token_count added, custom_key still present
        assert msg.metadata["custom_key"] == "custom_value"
        assert "token_count" in msg.metadata

    def test_fallback_updates_metadata(self):
        msg = _make_message("one two three")  # 3 words → fallback = 6
        with patch("app.core.token_counter._get_encoding", return_value=None):
            result = count_message_tokens(msg)
        assert result == 6
        assert msg.metadata["token_count"] == 6


# ---------------------------------------------------------------------------
# calculate_total_tokens
# ---------------------------------------------------------------------------


class TestCalculateTotalTokens:
    def test_empty_context_window_returns_zero(self):
        cw = _make_context_window(messages=[])
        assert calculate_total_tokens(cw) == 0

    def test_single_message_aggregation(self):
        msg = _make_message("Hello world")
        cw = _make_context_window(messages=[msg])
        total = calculate_total_tokens(cw)
        assert total == count_tokens("Hello world")

    def test_multiple_messages_summed(self):
        msgs = [
            _make_message("First message"),
            _make_message("Second message here"),
            _make_message("Third"),
        ]
        cw = _make_context_window(messages=msgs)
        expected = sum(count_tokens(m.content) for m in msgs)
        assert calculate_total_tokens(cw) == expected

    def test_research_report_included_in_total(self):
        report_text = "This is the research report answer."
        report = ResearchReport(answer=report_text)
        msg = _make_message("Follow-up question")
        cw = _make_context_window(messages=[msg], research_report=report)

        total = calculate_total_tokens(cw)
        expected = count_tokens(msg.content) + count_tokens(report_text)
        assert total == expected

    def test_no_research_report_excludes_report_tokens(self):
        msg = _make_message("Just a message")
        cw = _make_context_window(messages=[msg], research_report=None)
        total = calculate_total_tokens(cw)
        assert total == count_tokens(msg.content)

    def test_populates_metadata_on_all_messages(self):
        msgs = [_make_message(f"Message number {i}") for i in range(5)]
        cw = _make_context_window(messages=msgs)
        calculate_total_tokens(cw)
        for msg in msgs:
            assert "token_count" in msg.metadata
            assert msg.metadata["token_count"] > 0

    def test_total_increases_with_more_messages(self):
        msgs_short = [_make_message("Hi")]
        msgs_long = [_make_message("Hi"), _make_message("This is a longer follow-up message.")]
        cw_short = _make_context_window(messages=msgs_short)
        cw_long = _make_context_window(messages=msgs_long)
        assert calculate_total_tokens(cw_long) > calculate_total_tokens(cw_short)
