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


class SequencedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.call_count = 0

    def invoke(self, _messages):
        self.call_count += 1
        return SimpleNamespace(content=next(self.responses))


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
    assert "Citation-Bound Source Evidence" in human_context
    assert result.draft_answer == "Answer with [PDF-1]."


def test_chatbot_writer_does_not_crash_without_vector_context():
    llm = CapturingLLM()
    state = AgentState(question="Summarize the paper", external_context=[], vector_context=[])

    result = WriterAgent(llm).run(state)

    assert result.draft_answer == "Answer with [PDF-1]."
    assert "PDF / Vector Context" not in llm.messages[1].content


def test_writer_retries_once_when_model_returns_empty_content():
    llm = SequencedLLM(["   ", "Grounded answer [1]."])
    state = AgentState(
        question="Summarize the evidence",
        external_context=[{
            "title": "Evidence",
            "url": "https://example.com/evidence",
            "content": "Grounded evidence.",
        }],
    )

    result = WriterAgent(llm).run(state)

    assert llm.call_count == 2
    assert result.draft_answer.startswith("Grounded answer [1].\n\n## References\n[1]")
    assert result.failure_code is None


def test_writer_replaces_model_references_with_canonical_cited_entries():
    llm = SequencedLLM([
        "Grounded evidence [1].\n\n## References\n[99] Invented reference."
    ])
    state = AgentState(
        question="Summarize the evidence",
        external_context=[{
            "title": "Canonical Evidence",
            "url": "https://example.com/evidence",
            "content": "Grounded evidence.",
        }],
    )

    result = WriterAgent(llm).run(state)

    assert "[1] Canonical Evidence." in result.draft_answer
    assert "[99]" not in result.draft_answer


def test_writer_retries_evidence_sentinel_when_sources_are_available():
    llm = SequencedLLM(["Evidence not found in sources.", "Grounded report [1]."])
    state = AgentState(
        question="How does feedback improve reports?",
        external_context=[{
            "title": "Feedback study",
            "url": "https://example.com/feedback",
            "content": "Feedback improves revision quality.",
        }],
    )

    result = WriterAgent(llm).run(state)

    assert llm.call_count == 2
    assert result.draft_answer.startswith("Grounded report [1].")


def test_writer_replaces_repeated_evidence_sentinel_with_source_draft():
    llm = SequencedLLM(["Evidence not found in sources.", "Evidence not found in sources."])
    state = AgentState(
        question="How does feedback improve reports?",
        external_context=[{
            "title": "Feedback study",
            "url": "https://example.com/feedback",
            "content": "Feedback improves revision quality.",
        }],
    )

    result = WriterAgent(llm).run(state)

    assert result.failure_code is None
    assert "Evidence not found in sources." not in result.draft_answer
    assert "## Methodology" in result.draft_answer
    assert "Feedback improves revision quality." in result.draft_answer


def test_writer_marks_failure_after_two_empty_model_responses():
    llm = SequencedLLM(["", "   "])
    state = AgentState(
        question="Summarize the evidence",
        external_context=[{
            "title": "Evidence",
            "url": "https://example.com/evidence",
            "content": "Grounded evidence.",
        }],
    )

    result = WriterAgent(llm).run(state)

    assert llm.call_count == 2
    assert result.draft_answer is None
    assert result.failure_code == "writer_empty_response"
    assert result.workflow_status == "failed"


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
