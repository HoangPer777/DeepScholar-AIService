import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("INTERNAL_SERVICE_KEY", "test-key")

from app.workflows import rag_workflow


class FakePlanner:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, state):
        state.focus_sections = ["methodology", "results"]
        state.search_queries = ["structure aware chunking"]
        return state


class FakeClarifier:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, state):
        return state


class FakeResearcher:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, state):
        state.external_context = [
            {"title": "__research_notes__", "content": "Research notes"},
            {"title": "External source", "url": "https://example.com", "content": "External context"},
        ]
        return state


class FakeReader:
    def run(self, state):
        assert state.focus_sections == ["methodology", "results"]
        state.vector_context = [
            {
                "content": "Method chunk",
                "chunk_id": 1,
                "distance": 0.1,
                "section": "methodology",
                "section_title": "Proposed Approach",
                "chunk_type": "section_text",
                "heading_path": ["Paper", "Proposed Approach"],
            },
            {
                "content": "Table result chunk",
                "chunk_id": 2,
                "distance": 0.2,
                "section": "results",
                "section_title": "Evaluation",
                "chunk_type": "table",
                "heading_path": ["Paper", "Evaluation"],
            },
        ]
        return state


class FakeWriter:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, state):
        assert state.vector_context[0]["section"] == "methodology"
        assert state.vector_context[1]["chunk_type"] == "table"
        assert state.external_context[0]["title"] == "__research_notes__"
        state.draft_answer = "Draft answer citing [PDF-1] and [PDF-2]."
        return state


class FakeReviewer:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, state):
        state.reviewed_answer = state.draft_answer
        state.confidence_score = 0.91
        state.review_feedback = "No feedback."
        return state


def test_deep_research_multi_agent_flow_preserves_chunking_v2_context(monkeypatch):
    monkeypatch.setattr(rag_workflow, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(rag_workflow, "ClarifierAgent", FakeClarifier)
    monkeypatch.setattr(rag_workflow, "ResearcherAgent", FakeResearcher)
    monkeypatch.setattr(rag_workflow, "ReaderAgent", FakeReader)
    monkeypatch.setattr(rag_workflow, "WriterAgent", FakeWriter)
    monkeypatch.setattr(rag_workflow, "ReviewerAgent", FakeReviewer)
    monkeypatch.setattr(rag_workflow, "get_safe_llm", lambda _role: None)

    result = rag_workflow.run_chat_workflow(
        question="Compare method and results",
        article_id=123,
        session_id=None,
    )

    assert result["focus_sections"] == ["methodology", "results"]
    assert result["vector_context"][0]["section"] == "methodology"
    assert result["vector_context"][1]["chunk_type"] == "table"
    assert result["external_context"][1]["title"] == "External source"
    assert result["reviewed_answer"] == "Draft answer citing [PDF-1] and [PDF-2]."
    assert result["confidence_score"] == 0.91
