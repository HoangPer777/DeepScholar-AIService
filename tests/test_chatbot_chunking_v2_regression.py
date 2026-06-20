import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("INTERNAL_SERVICE_KEY", "test-key")

from app.agents.reader import ReaderAgent
from app.agents.writer import WriterAgent
from app.workflows.states import AgentState


class CapturingLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="Answer with [PDF-1].")


def test_chatbot_reader_retrieves_chunks_by_article_id(monkeypatch):
    def fake_search(article_id, question, focus_sections, limit, timings):
        assert article_id == 77
        assert "method" in question.lower()
        assert focus_sections == ["methodology"]
        return [
            {
                "content": "The method parses sections before embedding.",
                "chunk_id": 1,
                "distance": 0.1,
                "section": "methodology",
                "section_title": "Proposed Approach",
                "chunk_type": "section_text",
                "heading_path": ["Paper", "Proposed Approach"],
            }
        ]

    monkeypatch.setattr("app.agents.reader.search_article_chunks", fake_search)
    state = AgentState(question="Explain the method", article_id=77, focus_sections=["methodology"])

    result = ReaderAgent().run(state)

    assert result.vector_context[0]["section"] == "methodology"
    assert result.vector_context[0]["chunk_type"] == "section_text"


def test_chatbot_writer_receives_pdf_context_with_section_citation():
    llm = CapturingLLM()
    state = AgentState(
        question="What does the table show?",
        external_context=[],
        vector_context=[
            {
                "content": "Structure-aware chunking reaches Precision@5 of 0.78.",
                "chunk_id": 2,
                "distance": 0.05,
                "section": "results",
                "section_title": "Evaluation",
                "chunk_type": "table",
                "heading_path": ["Paper", "Evaluation"],
            }
        ],
    )

    result = WriterAgent(llm).run(state)
    human_context = llm.messages[1].content

    assert "[PDF-1] Section: results" in human_context
    assert "Precision@5 of 0.78" in human_context
    assert result.draft_answer == "Answer with [PDF-1]."


def test_chatbot_writer_does_not_crash_without_vector_context():
    llm = CapturingLLM()
    state = AgentState(question="Summarize the paper", external_context=[], vector_context=[])

    result = WriterAgent(llm).run(state)

    assert result.draft_answer == "Answer with [PDF-1]."
    assert "PDF / Vector Context" not in llm.messages[1].content


def test_chatbot_result_question_can_use_table_chunk(monkeypatch):
    def fake_search(article_id, question, focus_sections, limit, timings):
        assert focus_sections == ["results"]
        return [
            {
                "content": "| Method | Precision@5 |\n| Structure-aware | 0.78 |",
                "chunk_id": 10,
                "distance": 0.02,
                "section": "results",
                "section_title": "Evaluation",
                "chunk_type": "table",
                "heading_path": ["Paper", "Evaluation"],
            }
        ]

    monkeypatch.setattr("app.agents.reader.search_article_chunks", fake_search)
    state = AgentState(question="What result is in the table?", article_id=77, focus_sections=["results"])

    result = ReaderAgent().run(state)

    assert result.vector_context[0]["chunk_type"] == "table"
    assert "Precision@5" in result.vector_context[0]["content"]
