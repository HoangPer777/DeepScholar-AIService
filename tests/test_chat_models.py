"""
Unit tests for the Pydantic models defined in app.schemas.chat_models.

Tests cover:
- SessionStatus and MessageRole enum values
- Session model defaults and validation
- SessionMetadata model
- Message model defaults and validation
- Source model validation
- ResearchReport model
- ContextWindow model

Requirements: 1.1, 2.1
"""

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.chat_models import (
    ContextWindow,
    Message,
    MessageRole,
    ResearchReport,
    Session,
    SessionMetadata,
    SessionStatus,
    Source,
)


# ---------------------------------------------------------------------------
# SessionStatus enum
# ---------------------------------------------------------------------------


class TestSessionStatus:
    def test_enum_values(self):
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.INACTIVE == "inactive"
        assert SessionStatus.ARCHIVED == "archived"

    def test_all_values_present(self):
        values = {s.value for s in SessionStatus}
        assert values == {"active", "inactive", "archived"}

    def test_string_coercion(self):
        """SessionStatus is a str-enum so it should compare equal to its string value."""
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus("inactive") == SessionStatus.INACTIVE


# ---------------------------------------------------------------------------
# MessageRole enum
# ---------------------------------------------------------------------------


class TestMessageRole:
    def test_enum_values(self):
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"

    def test_all_values_present(self):
        values = {r.value for r in MessageRole}
        assert values == {"user", "assistant"}

    def test_string_coercion(self):
        assert MessageRole("user") == MessageRole.USER
        assert MessageRole("assistant") == MessageRole.ASSISTANT


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------


class TestSession:
    def _make_session(self, **kwargs):
        defaults = {
            "user_id": "user-abc",
            "initial_query": "What is quantum entanglement?",
        }
        defaults.update(kwargs)
        return Session(**defaults)

    def test_default_session_id_is_uuid(self):
        session = self._make_session()
        # Should be parseable as a UUID
        parsed = uuid.UUID(session.session_id)
        assert str(parsed) == session.session_id

    def test_each_session_gets_unique_id(self):
        s1 = self._make_session()
        s2 = self._make_session()
        assert s1.session_id != s2.session_id

    def test_default_status_is_active(self):
        session = self._make_session()
        assert session.status == SessionStatus.ACTIVE

    def test_default_message_count_is_zero(self):
        session = self._make_session()
        assert session.message_count == 0

    def test_created_at_is_datetime(self):
        session = self._make_session()
        assert isinstance(session.created_at, datetime)

    def test_updated_at_is_datetime(self):
        session = self._make_session()
        assert isinstance(session.updated_at, datetime)

    def test_status_accepts_enum_values(self):
        for status in SessionStatus:
            session = self._make_session(status=status)
            assert session.status == status

    def test_status_accepts_string_values(self):
        session = self._make_session(status="inactive")
        assert session.status == SessionStatus.INACTIVE

    def test_message_count_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            self._make_session(message_count=-1)

    def test_initial_query_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            self._make_session(initial_query="")

    def test_explicit_session_id_is_preserved(self):
        custom_id = str(uuid.uuid4())
        session = self._make_session(session_id=custom_id)
        assert session.session_id == custom_id

    def test_stores_user_id(self):
        session = self._make_session(user_id="user-xyz")
        assert session.user_id == "user-xyz"

    def test_stores_initial_query(self):
        query = "Tell me about black holes"
        session = self._make_session(initial_query=query)
        assert session.initial_query == query


# ---------------------------------------------------------------------------
# SessionMetadata model
# ---------------------------------------------------------------------------


class TestSessionMetadata:
    def _make_metadata(self, **kwargs):
        defaults = {
            "session_id": str(uuid.uuid4()),
            "initial_query": "Some query",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status": SessionStatus.ACTIVE,
            "message_count": 5,
        }
        defaults.update(kwargs)
        return SessionMetadata(**defaults)

    def test_basic_construction(self):
        meta = self._make_metadata()
        assert meta.message_count == 5
        assert meta.status == SessionStatus.ACTIVE

    def test_status_string_coercion(self):
        meta = self._make_metadata(status="archived")
        assert meta.status == SessionStatus.ARCHIVED


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------


class TestMessage:
    def _make_message(self, **kwargs):
        defaults = {
            "session_id": str(uuid.uuid4()),
            "role": MessageRole.USER,
            "content": "Hello, what is the paper about?",
        }
        defaults.update(kwargs)
        return Message(**defaults)

    def test_default_message_id_is_uuid(self):
        msg = self._make_message()
        parsed = uuid.UUID(msg.message_id)
        assert str(parsed) == msg.message_id

    def test_each_message_gets_unique_id(self):
        m1 = self._make_message()
        m2 = self._make_message()
        assert m1.message_id != m2.message_id

    def test_default_token_count_is_zero(self):
        msg = self._make_message()
        assert msg.token_count == 0

    def test_default_metadata_is_empty_dict(self):
        msg = self._make_message()
        assert msg.metadata == {}

    def test_timestamp_is_datetime(self):
        msg = self._make_message()
        assert isinstance(msg.timestamp, datetime)

    def test_role_user(self):
        msg = self._make_message(role=MessageRole.USER)
        assert msg.role == MessageRole.USER

    def test_role_assistant(self):
        msg = self._make_message(role=MessageRole.ASSISTANT)
        assert msg.role == MessageRole.ASSISTANT

    def test_role_string_coercion(self):
        msg = self._make_message(role="assistant")
        assert msg.role == MessageRole.ASSISTANT

    def test_content_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            self._make_message(content="")

    def test_token_count_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            self._make_message(token_count=-5)

    def test_metadata_stores_arbitrary_data(self):
        meta = {"sources": [1, 2], "confidence_score": 0.9, "is_summary": False}
        msg = self._make_message(metadata=meta)
        assert msg.metadata["confidence_score"] == 0.9
        assert msg.metadata["is_summary"] is False

    def test_stores_session_id(self):
        sid = str(uuid.uuid4())
        msg = self._make_message(session_id=sid)
        assert msg.session_id == sid


# ---------------------------------------------------------------------------
# Source model
# ---------------------------------------------------------------------------


class TestSource:
    def _make_source(self, **kwargs):
        defaults = {
            "index": 1,
            "type": "arxiv",
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "arxiv_id": "1706.03762",
        }
        defaults.update(kwargs)
        return Source(**defaults)

    def test_basic_construction(self):
        src = self._make_source()
        assert src.index == 1
        assert src.type == "arxiv"
        assert src.arxiv_id == "1706.03762"

    def test_index_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            self._make_source(index=0)

    def test_negative_index_rejected(self):
        with pytest.raises(ValidationError):
            self._make_source(index=-3)

    def test_url_source_type(self):
        src = self._make_source(type="url", arxiv_id=None)
        assert src.type == "url"
        assert src.arxiv_id is None

    def test_optional_fields_default_to_none(self):
        src = self._make_source()
        assert src.year is None
        assert src.venue is None
        assert src.abstract is None
        assert src.citation_count is None

    def test_authors_default_to_empty_list(self):
        src = self._make_source()
        assert src.authors == []

    def test_authors_stored_correctly(self):
        src = self._make_source(authors=["Vaswani", "Shazeer"])
        assert "Vaswani" in src.authors


# ---------------------------------------------------------------------------
# ResearchReport model
# ---------------------------------------------------------------------------


class TestResearchReport:
    def _make_report(self, **kwargs):
        defaults = {
            "answer": "Quantum entanglement is a phenomenon where two particles...",
        }
        defaults.update(kwargs)
        return ResearchReport(**defaults)

    def test_basic_construction(self):
        report = self._make_report()
        assert "Quantum" in report.answer

    def test_default_sources_is_empty_list(self):
        report = self._make_report()
        assert report.sources == []

    def test_default_confidence_score_is_zero(self):
        report = self._make_report()
        assert report.confidence_score == 0.0

    def test_default_review_feedback_is_empty_string(self):
        report = self._make_report()
        assert report.review_feedback == ""

    def test_default_planner_decision_is_empty_dict(self):
        report = self._make_report()
        assert report.planner_decision == {}

    def test_confidence_score_must_be_between_0_and_1(self):
        with pytest.raises(ValidationError):
            self._make_report(confidence_score=1.5)
        with pytest.raises(ValidationError):
            self._make_report(confidence_score=-0.1)

    def test_confidence_score_boundary_values(self):
        r0 = self._make_report(confidence_score=0.0)
        r1 = self._make_report(confidence_score=1.0)
        assert r0.confidence_score == 0.0
        assert r1.confidence_score == 1.0

    def test_answer_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            self._make_report(answer="")

    def test_sources_stored_correctly(self):
        src = Source(index=1, type="arxiv", title="Test Paper", url="https://example.com")
        report = self._make_report(sources=[src])
        assert len(report.sources) == 1
        assert report.sources[0].index == 1

    def test_planner_decision_stores_arbitrary_data(self):
        decision = {"strategy": "full_research", "reasoning": "New topic"}
        report = self._make_report(planner_decision=decision)
        assert report.planner_decision["strategy"] == "full_research"


# ---------------------------------------------------------------------------
# ContextWindow model
# ---------------------------------------------------------------------------


class TestContextWindow:
    def _make_context(self, **kwargs):
        defaults = {
            "session_id": str(uuid.uuid4()),
        }
        defaults.update(kwargs)
        return ContextWindow(**defaults)

    def test_basic_construction(self):
        ctx = self._make_context()
        assert ctx.messages == []
        assert ctx.sources == []
        assert ctx.total_tokens == 0
        assert ctx.is_compressed is False
        assert ctx.research_report is None

    def test_stores_session_id(self):
        sid = str(uuid.uuid4())
        ctx = self._make_context(session_id=sid)
        assert ctx.session_id == sid

    def test_total_tokens_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            self._make_context(total_tokens=-1)

    def test_is_compressed_defaults_to_false(self):
        ctx = self._make_context()
        assert ctx.is_compressed is False

    def test_is_compressed_can_be_set_true(self):
        ctx = self._make_context(is_compressed=True)
        assert ctx.is_compressed is True

    def test_messages_stored_correctly(self):
        sid = str(uuid.uuid4())
        msg = Message(session_id=sid, role=MessageRole.USER, content="Hello")
        ctx = self._make_context(session_id=sid, messages=[msg])
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == MessageRole.USER

    def test_research_report_stored_correctly(self):
        report = ResearchReport(answer="Some answer")
        ctx = self._make_context(research_report=report)
        assert ctx.research_report is not None
        assert ctx.research_report.answer == "Some answer"

    def test_sources_stored_correctly(self):
        src = Source(index=1, type="url", title="Example", url="https://example.com")
        ctx = self._make_context(sources=[src])
        assert len(ctx.sources) == 1
        assert ctx.sources[0].index == 1

    def test_total_tokens_reflects_sum(self):
        sid = str(uuid.uuid4())
        m1 = Message(session_id=sid, role=MessageRole.USER, content="Hi", token_count=10)
        m2 = Message(session_id=sid, role=MessageRole.ASSISTANT, content="Hello", token_count=20)
        # total_tokens is set explicitly by the caller (MemoryStore calculates it)
        ctx = self._make_context(session_id=sid, messages=[m1, m2], total_tokens=30)
        assert ctx.total_tokens == 30
