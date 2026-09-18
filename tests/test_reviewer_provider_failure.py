from unittest.mock import MagicMock, patch

import fakeredis
import pytest

from app.agents.reviewer import ReviewerAgent
from app.core.safe_llm import AllLLMProvidersFailed, SafeLLM
from app.workflows.states import AgentState


def test_safe_llm_wraps_groq_failure_after_openrouter_failures():
    groq = MagicMock()
    groq.invoke.side_effect = Exception("401 Unauthorized")
    llm = SafeLLM("reviewer", ["unavailable-model"], groq)

    openrouter = MagicMock()
    openrouter.invoke.side_effect = Exception("404 model unavailable")

    with (
        patch("app.core.safe_llm.settings") as mock_settings,
        patch.object(llm, "_build_openrouter_llm", return_value=openrouter),
        patch("app.core.safe_llm.time.sleep"),
    ):
        mock_settings.OPENROUTER_API_KEY = "test-key"
        with pytest.raises(AllLLMProvidersFailed):
            llm.invoke([{"role": "user", "content": "review"}])


def test_reviewer_rejects_when_all_providers_fail():
    llm = MagicMock()
    llm.invoke.side_effect = AllLLMProvidersFailed("all providers failed")
    state = AgentState(
        question="What is retrieval-augmented generation?",
        draft_answer="Best available draft with citations [1].",
        need_external_search=True,
        external_context=[
            {
                "title": "Academic source",
                "source_type": "arxiv",
                "url": "https://arxiv.org/abs/1234.5678",
                "citation_count": 50,
            }
        ],
    )

    result = ReviewerAgent(llm).run(state)
    assert result.reviewed_answer is None
    assert result.review_decision == "rejected"
    assert result.failure_code == "reviewer_unavailable"


def test_quality_rejection_returns_unreviewed_draft_without_failing_job():
    llm = MagicMock()
    llm.invoke.return_value.content = """{
        "score": 0.45,
        "decision": "rewrite",
        "failed_criteria": ["missing inline citations"],
        "feedback": "Add inline citations and a reference list."
    }"""
    state = AgentState(
        question="What is retrieval-augmented generation?",
        draft_answer="Draft without enough citations.",
        need_external_search=True,
        external_context=[
            {
                "title": "Academic source",
                "source_type": "arxiv",
                "url": "https://arxiv.org/abs/1234.5678",
                "citation_count": 50,
            }
        ],
    )

    result = ReviewerAgent(llm).run(state)

    assert result.draft_answer == "Draft without enough citations."
    assert result.reviewed_answer is None
    assert result.confidence_score == pytest.approx(0.45)
    assert result.iteration_count == 1
    assert result.review_decision == "rewrite"


def test_reviewer_rewrites_unsupported_named_identifier_even_if_llm_accepts():
    llm = MagicMock()
    llm.invoke.return_value.content = """{
        "score": 0.95,
        "decision": "accept",
        "failed_criteria": [],
        "feedback": "Looks grounded."
    }"""
    state = AgentState(
        question="How is retrieval used?",
        draft_answer="ChemPrompt and AgentFact improve retrieval reliability [1].",
        need_external_search=True,
        external_context=[
            {
                "title": "Retrieval study",
                "source_type": "arxiv",
                "url": "https://arxiv.org/abs/1234.5678",
                "citation_count": 50,
                "content": "This study evaluates standard retrieval-augmented generation.",
            }
        ],
    )

    result = ReviewerAgent(llm).run(state)

    assert result.reviewed_answer is None
    assert result.review_decision == "rewrite"
    assert result.confidence_score == pytest.approx(0.68)
    assert "ChemPrompt" in result.review_feedback
    assert "AgentFact" in result.review_feedback
    reviewer_context = llm.invoke.call_args.args[0][1].content
    assert "Citation-Bound Evidence" in reviewer_context
    assert "standard retrieval-augmented generation" in reviewer_context


def test_reviewer_accepts_named_identifier_present_in_cited_evidence():
    llm = MagicMock()
    llm.invoke.return_value.content = """{
        "score": 0.91,
        "decision": "accept",
        "failed_criteria": [],
        "feedback": "Grounded."
    }"""
    state = AgentState(
        question="How is retrieval used?",
        draft_answer=(
            "ChemPrompt uses retrieved molecular evidence to guide generation [1].\n\n"
            "## References\n[1] ChemPrompt study."
        ),
        need_external_search=True,
        external_context=[
            {
                "title": "ChemPrompt study",
                "source_type": "arxiv",
                "url": "https://arxiv.org/abs/1234.5678",
                "citation_count": 50,
                "content": "ChemPrompt uses retrieved molecular evidence to guide generation.",
            }
        ],
    )

    result = ReviewerAgent(llm).run(state)

    assert result.reviewed_answer == state.draft_answer
    assert result.review_decision == "accept"


def test_reviewer_does_not_treat_generic_acronyms_as_named_systems():
    llm = MagicMock()
    llm.invoke.return_value.content = """{
        "score": 0.91,
        "decision": "accept",
        "failed_criteria": [],
        "feedback": "Grounded."
    }"""
    state = AgentState(
        question="How are metadata sources used?",
        draft_answer=(
            "DOI and NOAA metadata can support scholarly indexing [1].\n\n"
            "## References\n[1] Metadata study."
        ),
        need_external_search=True,
        external_context=[
            {
                "title": "Metadata study",
                "source_type": "arxiv",
                "url": "https://arxiv.org/abs/1234.5678",
                "citation_count": 50,
                "content": "This study discusses scholarly indexing metadata.",
            }
        ],
    )

    result = ReviewerAgent(llm).run(state)

    assert result.reviewed_answer == state.draft_answer
    assert result.review_decision == "accept"


def test_reviewer_skips_llm_when_writer_response_is_empty():
    llm = MagicMock()
    state = AgentState(
        question="How is retrieval used?",
        draft_answer=None,
        failure_code="writer_empty_response",
        failure_message="Writer model returned an empty response after one retry.",
    )

    result = ReviewerAgent(llm).run(state)

    llm.invoke.assert_not_called()
    assert result.reviewed_answer is None
    assert result.review_decision == "rejected"
    assert result.confidence_score == 0.0
    assert result.review_feedback == "Writer model returned an empty response after one retry."


def test_reviewer_skips_sentinel_draft_as_unusable_writer_output():
    llm = MagicMock()
    state = AgentState(
        question="How does feedback improve reports?",
        draft_answer="Evidence not found in sources.",
        failure_code="writer_insufficient_evidence",
        failure_message="Writer could not find usable evidence excerpts for this question.",
    )

    result = ReviewerAgent(llm).run(state)

    llm.invoke.assert_not_called()
    assert result.reviewed_answer is None
    assert result.review_decision == "rejected"
    assert result.review_feedback == state.failure_message


def test_reviewer_rewrites_citation_without_citable_evidence():
    llm = MagicMock()
    llm.invoke.return_value.content = """{
        "score": 0.9,
        "decision": "accept",
        "failed_criteria": [],
        "feedback": "Looks grounded."
    }"""
    sources = [
        {
            "title": f"Source {index}",
            "source_type": "arxiv",
            "url": f"https://example.com/{index}",
            "citation_count": 50,
            "content": f"Evidence {index}",
        }
        for index in range(1, 29)
    ]
    state = AgentState(
        question="How does verification work?",
        draft_answer="A later source supports this claim [28].\n\n## References\n[28] Source 28.",
        need_external_search=True,
        external_context=sources,
    )

    result = ReviewerAgent(llm).run(state)

    assert result.reviewed_answer is None
    assert result.review_decision == "rewrite"
    assert "citation_without_evidence:[28]" in result.review_feedback


def test_reviewer_rewrites_citation_missing_reference_entry():
    llm = MagicMock()
    llm.invoke.return_value.content = """{
        "score": 0.9,
        "decision": "accept",
        "failed_criteria": [],
        "feedback": "Looks grounded."
    }"""
    state = AgentState(
        question="How does verification work?",
        draft_answer="Verification improves citation traceability [1].",
        need_external_search=True,
        external_context=[{
            "title": "Verification study",
            "source_type": "arxiv",
            "url": "https://example.com/verification",
            "citation_count": 50,
            "content": "Verification improves citation traceability.",
        }],
    )

    result = ReviewerAgent(llm).run(state)

    assert result.reviewed_answer is None
    assert result.review_decision == "rewrite"
    assert "citation_missing_reference:[1]" in result.review_feedback


def test_reviewer_rewrites_unsupported_percentage_claim():
    llm = MagicMock()
    llm.invoke.return_value.content = """{
        "score": 0.9,
        "decision": "accept",
        "failed_criteria": [],
        "feedback": "Looks grounded."
    }"""
    state = AgentState(
        question="How much does verification improve precision?",
        draft_answer=(
            "Verification improves precision by 12% [1].\n\n"
            "## References\n[1] Verification study."
        ),
        need_external_search=True,
        external_context=[{
            "title": "Verification study",
            "source_type": "arxiv",
            "url": "https://example.com/verification",
            "citation_count": 50,
            "content": "Verification improves citation traceability.",
        }],
    )

    result = ReviewerAgent(llm).run(state)

    assert result.reviewed_answer is None
    assert result.review_decision == "rewrite"
    assert "12%" in result.review_feedback


def test_reviewer_accepts_percentage_present_in_cited_evidence():
    llm = MagicMock()
    llm.invoke.return_value.content = """{
        "score": 0.9,
        "decision": "accept",
        "failed_criteria": [],
        "feedback": "Grounded."
    }"""
    draft = "Precision improves by 12% [1].\n\n## References\n[1] Verification study."
    state = AgentState(
        question="How much does verification improve precision?",
        draft_answer=draft,
        need_external_search=True,
        external_context=[{
            "title": "Verification study",
            "source_type": "arxiv",
            "url": "https://example.com/verification",
            "citation_count": 50,
            "content": "The evaluation reports that precision improves by 12%.",
        }],
    )

    result = ReviewerAgent(llm).run(state)

    assert result.reviewed_answer == draft
    assert result.review_decision == "accept"


def test_rejected_draft_is_not_saved_as_fast_chat_context():
    from app.workflows.rag_workflow import _save_research_context

    redis_client = MagicMock()
    store = MagicMock()
    result = {
        "reviewed_answer": None,
        "draft_answer": "Draft retained for follow-up questions.",
        "confidence_score": 0.45,
        "review_feedback": "Add inline citations.",
        "external_context": [],
    }

    with (
        patch("app.workflows.rag_workflow.create_redis_client", return_value=redis_client),
        patch("app.workflows.rag_workflow.MemoryStore", return_value=store),
    ):
        _save_research_context("session-1", result)

    store.init_session_context.assert_not_called()
    redis_client.close.assert_called_once()


def test_rejected_draft_does_not_create_fast_chat_context():
    from app.workflows.rag_workflow import _save_research_context

    server = fakeredis.FakeServer()

    def make_redis_client():
        return fakeredis.FakeRedis(server=server)

    initial_result = {
        "reviewed_answer": None,
        "draft_answer": "A rejected but usable RAG research draft.",
        "confidence_score": 0.45,
        "review_feedback": "Add inline citations.",
        "external_context": [],
    }

    with (
        patch("app.workflows.rag_workflow.create_redis_client", side_effect=make_redis_client),
    ):
        _save_research_context("session-1", initial_result)
        from app.core.memory_store import MemoryStore, SessionContextNotFoundError
        with pytest.raises(SessionContextNotFoundError):
            MemoryStore(make_redis_client()).get_context_window("session-1")


def test_fast_chat_llm_failure_does_not_fall_through_to_full_pipeline():
    from app.core.safe_llm import AllLLMProvidersFailed
    from app.workflows.rag_workflow import (
        _save_research_context,
        run_chat_workflow,
    )

    server = fakeredis.FakeServer()

    def make_redis_client():
        return fakeredis.FakeRedis(server=server)

    failed_fast_llm = MagicMock()
    failed_fast_llm.invoke.side_effect = AllLLMProvidersFailed("fast chat unavailable")
    initial_result = {
        "reviewed_answer": "Accepted research report.",
        "draft_answer": None,
        "confidence_score": 0.9,
        "review_feedback": "Accepted.",
        "external_context": [],
    }

    with (
        patch("app.workflows.rag_workflow.create_redis_client", side_effect=make_redis_client),
        patch("app.workflows.rag_workflow.get_safe_llm", return_value=failed_fast_llm) as get_llm,
    ):
        _save_research_context("session-1", initial_result)
        with pytest.raises(AllLLMProvidersFailed, match="fast chat unavailable"):
            run_chat_workflow(
                question="Explain it briefly.",
                session_id="session-1",
            )

    get_llm.assert_called_once_with("fast_chat")
    failed_fast_llm.invoke.assert_called_once()


def test_removed_unavailable_glm_free_slug():
    from app.core.model_candidates import MODEL_CANDIDATES

    candidates = [model for models in MODEL_CANDIDATES.values() for model in models]
    assert "z-ai/glm-4.5-air:free" not in candidates
    assert MODEL_CANDIDATES["fast_chat"]
