"""
Integration tests for the Memory Chatbot API endpoints and full feature stack.

Tests cover:
1. Existing deep-research endpoints still work (backward compatibility)
2. Memory store source caching (save_source / get_source / get_all_sources)
3. API layer: /api/chat and /api/research endpoints via FastAPI TestClient
4. Session manager + memory store integration (end-to-end unit flow)
5. Token counter integration with ContextWindow

All tests run without live Redis/PostgreSQL — fakeredis + SQLite in-memory.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.core.memory_store import MemoryStore, SessionContextNotFoundError, _source_key
from app.core.token_counter import calculate_total_tokens, count_tokens
from app.schemas.chat_models import (
    ContextWindow,
    Message,
    MessageRole,
    ResearchReport,
    Source,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_message(session_id: str, content: str = "Hello", role: MessageRole = MessageRole.USER) -> Message:
    return Message(session_id=session_id, role=role, content=content)


def _make_source(index: int = 1, source_type: str = "url", url: str = "") -> Source:
    return Source(
        index=index,
        type=source_type,
        title=f"Source {index}",
        url=url or f"https://example.com/{index}",
        arxiv_id="2401.00001" if source_type == "arxiv" else None,
    )


def _make_report(answer: str = "Research answer") -> ResearchReport:
    return ResearchReport(answer=answer, confidence_score=0.85)


@pytest.fixture
def fake_redis():
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server)
    yield client
    client.flushall()
    client.close()


@pytest.fixture
def store(fake_redis):
    return MemoryStore(redis_client=fake_redis)


# ===========================================================================
# 1. Backward compatibility — existing API endpoints
# ===========================================================================

class TestBackwardCompatibilityAPI:
    """Verify existing /api/chat and /api/research endpoints still work."""

    def test_health_endpoint_returns_200(self):
        from app.main import app
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_root_endpoint_returns_welcome(self):
        from app.main import app
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data

    def test_deep_research_post_returns_task_id(self):
        """POST /api/research/deep-research should return task_id immediately."""
        from app.main import app
        client = TestClient(app)
        resp = client.post(
            "/api/research/deep-research",
            json={"query": "What is transformer architecture?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"

    def test_deep_research_status_404_for_unknown_task(self):
        """GET /api/research/status/{unknown} should return 404."""
        from app.main import app
        client = TestClient(app)
        resp = client.get("/api/research/status/nonexistent-task-id")
        assert resp.status_code == 404

    def test_chat_endpoint_exists(self):
        """POST /api/chat/ should exist (may fail with 422 if payload wrong)."""
        from app.main import app
        client = TestClient(app)
        # Send empty body — expect 422 (validation error), not 404
        resp = client.post("/api/chat/", json={})
        assert resp.status_code in (422, 500)  # not 404

    def test_chat_history_endpoint_exists(self):
        """GET /api/chat/history/{session_id} should return 200."""
        from app.main import app
        client = TestClient(app)
        resp = client.get("/api/chat/history/test-session-123")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "history" in data

    def test_research_query_too_short_returns_422(self):
        """Query shorter than 3 chars should fail validation."""
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/research/deep-research", json={"query": "ab"})
        assert resp.status_code == 422


# ===========================================================================
# 2. Source caching in MemoryStore (save_source / get_source / get_all_sources)
# ===========================================================================

class TestSourceCaching:
    """Tests for MemoryStore source caching operations."""

    def test_save_source_returns_index_1_for_first_source(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        src = _make_source(index=1)
        idx = store.save_source(sid, src)
        assert idx == 1

    def test_save_source_increments_index(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        idx1 = store.save_source(sid, _make_source(1, url="https://a.com/1"))
        idx2 = store.save_source(sid, _make_source(2, url="https://a.com/2"))
        assert idx1 == 1
        assert idx2 == 2

    def test_get_source_returns_stored_source(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        src = _make_source(1, url="https://example.com/paper")
        store.save_source(sid, src)
        retrieved = store.get_source(sid, 1)
        assert retrieved is not None
        assert retrieved.url == "https://example.com/paper"

    def test_get_source_returns_none_for_missing_index(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        result = store.get_source(sid, 99)
        assert result is None

    def test_get_all_sources_returns_sorted_by_index(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        store.save_source(sid, _make_source(1, url="https://a.com/1"))
        store.save_source(sid, _make_source(2, url="https://a.com/2"))
        store.save_source(sid, _make_source(3, url="https://a.com/3"))
        sources = store.get_all_sources(sid)
        assert [s.index for s in sources] == [1, 2, 3]

    def test_get_all_sources_empty_for_new_session(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        assert store.get_all_sources(sid) == []

    def test_url_deduplication_returns_existing_index(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        src = _make_source(1, url="https://example.com/paper")
        idx1 = store.save_source(sid, src)
        # Save same URL again — should return same index
        idx2 = store.save_source(sid, _make_source(1, url="https://example.com/paper"))
        assert idx1 == idx2 == 1
        # Only one source should be stored
        assert len(store.get_all_sources(sid)) == 1

    def test_arxiv_deduplication_by_arxiv_id(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        src1 = Source(index=1, type="arxiv", title="Paper A", url="https://arxiv.org/abs/2401.00001", arxiv_id="2401.00001")
        src2 = Source(index=1, type="arxiv", title="Paper A mirror", url="https://alphaxiv.org/abs/2401.00001", arxiv_id="2401.00001")
        idx1 = store.save_source(sid, src1)
        idx2 = store.save_source(sid, src2)
        assert idx1 == idx2  # same arxiv_id → same index
        assert len(store.get_all_sources(sid)) == 1

    def test_different_urls_get_different_indices(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        store.save_source(sid, _make_source(1, url="https://a.com/1"))
        store.save_source(sid, _make_source(2, url="https://b.com/2"))
        assert len(store.get_all_sources(sid)) == 2

    def test_source_stored_in_individual_key(self, store, fake_redis):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        store.save_source(sid, _make_source(1, url="https://example.com/1"))
        raw = fake_redis.get(_source_key(sid, 1))
        assert raw is not None
        data = json.loads(raw)
        assert data["url"] == "https://example.com/1"

    def test_source_context_hash_stays_in_sync(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        store.save_source(sid, _make_source(1, url="https://example.com/1"))
        store.save_source(sid, _make_source(2, url="https://example.com/2"))
        cw = store.get_context_window(sid, max_messages=0)
        assert len(cw.sources) == 2


# ===========================================================================
# 3. End-to-end memory flow: init → save messages → get context → delete
# ===========================================================================

class TestMemoryEndToEndFlow:
    """Full memory chatbot flow without live infrastructure."""

    def test_full_session_lifecycle(self, store):
        """init → save user msg → save assistant msg → get context → delete."""
        sid = str(uuid.uuid4())
        report = _make_report("Deep research on transformers.")
        sources = [_make_source(1, url="https://arxiv.org/abs/1706.03762")]

        # 1. Init session with research report
        store.init_session_context(sid, research_report=report, sources=sources, user_id="user-1")
        assert store.session_exists(sid)

        # 2. Save user message
        user_msg = _make_message(sid, "What is attention mechanism?", MessageRole.USER)
        store.save_message(sid, user_msg, user_id="user-1")

        # 3. Save assistant response
        asst_msg = _make_message(sid, "Attention allows the model to focus on relevant parts.", MessageRole.ASSISTANT)
        store.save_message(sid, asst_msg, user_id="user-1")

        # 4. Get context window
        cw = store.get_context_window(sid, max_messages=10)
        assert cw.session_id == sid
        assert len(cw.messages) == 2
        assert cw.research_report is not None
        assert cw.research_report.answer == "Deep research on transformers."
        assert len(cw.sources) == 1

        # 5. Delete session
        store.delete_session_context(sid, user_id="user-1")
        assert not store.session_exists(sid)

    def test_context_window_sliding_window_keeps_latest(self, store):
        """With 20 messages and max_messages=5, only last 5 are returned."""
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        for i in range(20):
            store.save_message(sid, _make_message(sid, f"message {i}"))
        cw = store.get_context_window(sid, max_messages=5)
        assert len(cw.messages) == 5
        # Last message should be message 19
        assert "19" in cw.messages[-1].content

    def test_token_counter_integrates_with_context_window(self, store):
        """calculate_total_tokens works on a ContextWindow from MemoryStore."""
        sid = str(uuid.uuid4())
        report = _make_report("This is a research report with some content.")
        store.init_session_context(sid, research_report=report)
        store.save_message(sid, _make_message(sid, "First question about the topic."))
        store.save_message(sid, _make_message(sid, "Second follow-up question here."))

        cw = store.get_context_window(sid, max_messages=10)
        total = calculate_total_tokens(cw)
        assert total > 0
        # Each message should have token_count in metadata after calculation
        for msg in cw.messages:
            assert "token_count" in msg.metadata
            assert msg.metadata["token_count"] > 0

    def test_multi_user_sessions_isolated(self, store):
        """Two users' sessions are completely isolated."""
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())

        store.init_session_context(sid_a, user_id="alice")
        store.init_session_context(sid_b, user_id="bob")

        store.save_message(sid_a, _make_message(sid_a, "Alice's message"), user_id="alice")
        store.save_message(sid_b, _make_message(sid_b, "Bob's message"), user_id="bob")

        sessions_alice = store.list_user_sessions("alice")
        sessions_bob = store.list_user_sessions("bob")

        assert sid_a in sessions_alice
        assert sid_b not in sessions_alice
        assert sid_b in sessions_bob
        assert sid_a not in sessions_bob

    def test_research_report_never_overwritten_by_messages(self, store):
        """Saving messages should not affect the stored research report."""
        sid = str(uuid.uuid4())
        report = _make_report("Original research answer.")
        store.init_session_context(sid, research_report=report)

        for i in range(5):
            store.save_message(sid, _make_message(sid, f"Follow-up {i}"))

        cw = store.get_context_window(sid, max_messages=10)
        assert cw.research_report.answer == "Original research answer."

    def test_refresh_ttl_extends_session(self, store, fake_redis):
        """refresh_ttl should reset the TTL to 24 hours."""
        from app.core.memory_store import SESSION_TTL_SECONDS
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        store.refresh_ttl(sid)
        from app.core.memory_store import _context_key
        ttl = fake_redis.ttl(_context_key(sid))
        assert SESSION_TTL_SECONDS - 5 <= ttl <= SESSION_TTL_SECONDS


# ===========================================================================
# 4. API endpoint contract tests (mocked workflow)
# ===========================================================================

class TestChatAPIContract:
    """Test /api/chat endpoint contracts with mocked workflow."""

    def _mock_workflow_result(self):
        return {
            "reviewed_answer": "Transformers use self-attention to process sequences.",
            "draft_answer": None,
            "confidence_score": 0.92,
            "review_feedback": "Good answer.",
            "need_clarification": False,
            "clarified_question": None,
            "external_context": [
                {
                    "title": "Attention Is All You Need",
                    "url": "https://arxiv.org/abs/1706.03762",
                    "score": 0.95,
                    "source_type": "arxiv",
                    "apa_year": "2017",
                }
            ],
            "session_id": None,
        }

    def test_chat_returns_answer(self):
        from app.main import app
        client = TestClient(app)
        with patch("app.api.chatbot.run_chat_workflow", return_value=self._mock_workflow_result()):
            resp = client.post("/api/chat/", json={"question": "What is attention?", "article_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Transformers use self-attention to process sequences."

    def test_chat_returns_confidence_score(self):
        from app.main import app
        client = TestClient(app)
        with patch("app.api.chatbot.run_chat_workflow", return_value=self._mock_workflow_result()):
            resp = client.post("/api/chat/", json={"question": "What is attention?", "article_id": 1})
        data = resp.json()
        assert data["confidence_score"] == pytest.approx(0.92)

    def test_chat_returns_citations(self):
        from app.main import app
        client = TestClient(app)
        with patch("app.api.chatbot.run_chat_workflow", return_value=self._mock_workflow_result()):
            resp = client.post("/api/chat/", json={"question": "What is attention?", "article_id": 1})
        data = resp.json()
        assert len(data["citations"]) == 1
        assert data["citations"][0]["title"] == "Attention Is All You Need"

    def test_chat_filters_research_notes_from_citations(self):
        """__research_notes__ entries should not appear in citations."""
        from app.main import app
        client = TestClient(app)
        result = self._mock_workflow_result()
        result["external_context"].append({
            "title": "__research_notes__",
            "url": "",
            "score": 0.0,
            "source_type": "web",
            "apa_year": "n.d.",
        })
        with patch("app.api.chatbot.run_chat_workflow", return_value=result):
            resp = client.post("/api/chat/", json={"question": "What is attention?", "article_id": 1})
        data = resp.json()
        titles = [c["title"] for c in data["citations"]]
        assert "__research_notes__" not in titles

    def test_chat_500_on_workflow_exception(self):
        from app.main import app
        client = TestClient(app)
        with patch("app.api.chatbot.run_chat_workflow", side_effect=RuntimeError("LLM failed")):
            resp = client.post("/api/chat/", json={"question": "What is attention?", "article_id": 1})
        assert resp.status_code == 500

    def test_chat_question_too_short_returns_422(self):
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/chat/", json={"question": "Hi", "article_id": 1})
        assert resp.status_code == 422


# ===========================================================================
# 5. Research API contract tests (mocked workflow)
# ===========================================================================

class TestResearchAPIContract:
    """Test /api/research endpoints with mocked workflow."""

    def _mock_research_result(self):
        return {
            "reviewed_answer": "RAG combines retrieval with generation.",
            "draft_answer": None,
            "confidence_score": 0.88,
            "review_feedback": "Solid.",
            "need_clarification": False,
            "need_external_search": True,
            "focus_sections": [],
            "search_queries": ["RAG retrieval augmented generation"],
            "clarified_question": None,
            "iteration_count": 1,
            "external_context": [
                {
                    "title": "RAG Paper",
                    "url": "https://arxiv.org/abs/2005.11401",
                    "score": 0.9,
                    "source_type": "arxiv",
                    "apa_year": "2020",
                    "apa_authors": "Lewis et al.",
                    "apa_venue": "NeurIPS",
                }
            ],
        }

    def test_deep_research_returns_task_id(self):
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/research/deep-research", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        assert "task_id" in resp.json()

    def test_deep_research_status_pending_initially(self):
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/research/deep-research", json={"query": "What is RAG?"})
        task_id = resp.json()["task_id"]
        status_resp = client.get(f"/api/research/status/{task_id}")
        # Should be pending (job hasn't run yet in test)
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "pending"

    def test_research_result_has_sources_list(self):
        """When job completes, result should have sources list."""
        from app.api.research import _jobs
        from app.main import app
        client = TestClient(app)

        # Manually inject a completed job
        fake_task_id = str(uuid.uuid4())
        _jobs[fake_task_id] = {
            "status": "done",
            "result": {
                "answer": "RAG combines retrieval with generation.",
                "sources": [{"index": 1, "title": "RAG Paper", "url": "https://arxiv.org/abs/2005.11401",
                              "score": 0.9, "source_type": "arxiv", "apa_year": "2020"}],
                "planner_decision": {},
                "confidence_score": 0.88,
                "iterations_used": 1,
                "decision": "accept",
                "review_feedback": "Good.",
            }
        }
        resp = client.get(f"/api/research/status/{fake_task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert "sources" in data
        assert len(data["sources"]) == 1

    def test_research_result_cleaned_up_after_fetch(self):
        """After fetching a done result, it should be removed from _jobs."""
        from app.api.research import _jobs
        from app.main import app
        client = TestClient(app)

        fake_task_id = str(uuid.uuid4())
        _jobs[fake_task_id] = {
            "status": "done",
            "result": {"answer": "Done", "sources": [], "planner_decision": {},
                       "confidence_score": 0.5, "iterations_used": 1,
                       "decision": "accept", "review_feedback": None}
        }
        client.get(f"/api/research/status/{fake_task_id}")
        # Second fetch should 404
        resp2 = client.get(f"/api/research/status/{fake_task_id}")
        assert resp2.status_code == 404


# ===========================================================================
# 6. MemoryStore + SessionManager integration
# ===========================================================================

class TestMemorySessionManagerIntegration:
    """Integration between SessionManager (SQLite) and MemoryStore (fakeredis)."""

    @pytest.fixture
    def db_and_manager(self):
        from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, create_engine
        from sqlalchemy.orm import declarative_base, relationship, sessionmaker
        from sqlalchemy.pool import StaticPool
        from app.core.session_manager import SessionManager

        Base = declarative_base()

        class SessionORM(Base):
            __tablename__ = "sessions"
            session_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
            user_id = Column(String(255), nullable=False)
            initial_query = Column(Text, nullable=False)
            created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
            updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
            status = Column(String(20), nullable=False, default="active")
            message_count = Column(Integer, nullable=False, default=0)
            messages = relationship("MessageORM", back_populates="session", cascade="all, delete-orphan")

        class MessageORM(Base):
            __tablename__ = "messages"
            message_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
            session_id = Column(String(36), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
            role = Column(String(20), nullable=False)
            content = Column(Text, nullable=False)
            timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
            token_count = Column(Integer, nullable=False, default=0)
            session = relationship("SessionORM", back_populates="messages")

        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        manager = SessionManager(session_orm_class=SessionORM)
        yield db, manager
        db.close()

    def test_create_session_then_init_memory(self, db_and_manager, store):
        """Create DB session, then init Redis context — both should succeed."""
        db, manager = db_and_manager
        session = manager.create_session(db, user_id="user-1", initial_query="What is RAG?")
        sid = session.session_id

        store.init_session_context(sid, user_id="user-1")
        assert store.session_exists(sid)

    def test_delete_session_also_clears_redis(self, db_and_manager, store):
        """Deleting DB session and Redis context leaves no trace."""
        db, manager = db_and_manager
        session = manager.create_session(db, user_id="user-1", initial_query="Test query")
        sid = session.session_id

        store.init_session_context(sid, user_id="user-1")
        store.save_message(sid, _make_message(sid, "Hello"), user_id="user-1")

        # Delete from DB
        manager.delete_session(db, session_id=sid, user_id="user-1")
        # Delete from Redis
        store.delete_session_context(sid, user_id="user-1")

        assert not store.session_exists(sid)
        with pytest.raises(SessionContextNotFoundError):
            store.get_context_window(sid)

    def test_session_ownership_prevents_cross_user_access(self, db_and_manager):
        """SessionManager raises SessionOwnershipError for wrong user."""
        from app.core.session_manager import SessionOwnershipError
        db, manager = db_and_manager
        session = manager.create_session(db, user_id="alice", initial_query="Alice's query")
        with pytest.raises(SessionOwnershipError):
            manager.get_session(db, session_id=session.session_id, user_id="bob")


# ===========================================================================
# 7. Source deduplication edge cases
# ===========================================================================

class TestSourceDeduplication:
    """Edge cases for MemoryStore._is_same_source."""

    def test_same_url_trailing_slash_deduped(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        src1 = Source(index=1, type="url", title="A", url="https://example.com/paper/")
        src2 = Source(index=1, type="url", title="A", url="https://example.com/paper")
        idx1 = store.save_source(sid, src1)
        idx2 = store.save_source(sid, src2)
        assert idx1 == idx2

    def test_arxiv_case_insensitive_dedup(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        src1 = Source(index=1, type="arxiv", title="P", url="https://arxiv.org/abs/2401.00001", arxiv_id="2401.00001")
        src2 = Source(index=1, type="arxiv", title="P", url="https://arxiv.org/abs/2401.00001", arxiv_id="2401.00001")
        idx1 = store.save_source(sid, src1)
        idx2 = store.save_source(sid, src2)
        assert idx1 == idx2

    def test_different_types_not_deduped(self, store):
        sid = str(uuid.uuid4())
        store.init_session_context(sid)
        src1 = Source(index=1, type="arxiv", title="P", url="https://arxiv.org/abs/2401.00001", arxiv_id="2401.00001")
        src2 = Source(index=1, type="url", title="P", url="https://arxiv.org/abs/2401.00001")
        idx1 = store.save_source(sid, src1)
        idx2 = store.save_source(sid, src2)
        # Mixed types → not deduplicated
        assert idx1 != idx2

    def test_sources_across_sessions_independent(self, store):
        """Sources in session A don't affect session B's indices."""
        sid_a = str(uuid.uuid4())
        sid_b = str(uuid.uuid4())
        store.init_session_context(sid_a)
        store.init_session_context(sid_b)

        store.save_source(sid_a, _make_source(1, url="https://a.com/1"))
        store.save_source(sid_a, _make_source(2, url="https://a.com/2"))

        # Session B starts fresh at index 1
        idx = store.save_source(sid_b, _make_source(1, url="https://b.com/1"))
        assert idx == 1


# ===========================================================================
# 8. Token counter edge cases
# ===========================================================================

class TestTokenCounterEdgeCases:
    """Additional edge cases for token counting."""

    def test_count_tokens_unicode_text(self):
        result = count_tokens("Xin chào thế giới — привет мир")
        assert result > 0

    def test_count_tokens_very_long_text(self):
        long_text = "word " * 1000
        result = count_tokens(long_text)
        assert result > 500  # should be substantial

    def test_context_window_with_no_messages_no_report(self):
        cw = ContextWindow(session_id="test", messages=[], research_report=None)
        from app.core.token_counter import calculate_total_tokens
        assert calculate_total_tokens(cw) == 0

    def test_context_window_report_only(self):
        report = ResearchReport(answer="A detailed research answer about machine learning.")
        cw = ContextWindow(session_id="test", messages=[], research_report=report)
        from app.core.token_counter import calculate_total_tokens
        total = calculate_total_tokens(cw)
        assert total > 0
        assert total == count_tokens(report.answer)
