import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("INTERNAL_SERVICE_KEY", "test-key")

from app.agents.reader import ReaderAgent
from app.workflows.states import AgentState


def test_reader_agent_preserves_vector_metadata(monkeypatch):
    def fake_search_article_chunks(article_id, question, focus_sections, limit, timings):
        assert article_id == 11
        assert focus_sections == ["methodology"]
        return [
            {
                "content": "method chunk",
                "chunk_id": 101,
                "distance": 0.12,
                "section": "methodology",
                "section_title": "Proposed Approach",
                "chunk_type": "section_text",
                "heading_path": ["Title", "Proposed Approach"],
                "page_start": 1,
                "page_end": 2,
                "chunk_index": 3,
            }
        ]

    monkeypatch.setattr("app.agents.reader.search_article_chunks", fake_search_article_chunks)
    state = AgentState(question="How does it work?", article_id=11, focus_sections=["methodology"])

    result = ReaderAgent().run(state)

    assert result.vector_context == [
        {
            "content": "method chunk",
            "chunk_id": 101,
            "distance": 0.12,
            "section": "methodology",
            "section_title": "Proposed Approach",
            "chunk_type": "section_text",
            "heading_path": ["Title", "Proposed Approach"],
            "page_start": 1,
            "page_end": 2,
            "chunk_index": 3,
        }
    ]


def test_reader_agent_keeps_old_failure_fallback(monkeypatch):
    def failing_search(*_args, **_kwargs):
        raise RuntimeError("vector db unavailable")

    monkeypatch.setattr("app.agents.reader.search_article_chunks", failing_search)
    state = AgentState(question="What is this?", article_id=11)

    result = ReaderAgent().run(state)

    assert result.vector_context == []
    assert any("vector search failed" in log for log in result.logs)
