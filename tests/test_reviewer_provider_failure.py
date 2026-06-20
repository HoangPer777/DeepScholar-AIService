from unittest.mock import MagicMock, patch

import fakeredis
import pytest

from app.agents.reviewer import ReviewRejectedError, ReviewerAgent
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

    with pytest.raises(ReviewRejectedError, match="reviewer unavailable"):
        ReviewerAgent(llm).run(state)

    assert state.reviewed_answer is None


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
    assert result.iteration_count == result.max_iterations


def test_rejected_draft_is_saved_as_fast_chat_context():
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

    saved_report = store.init_session_context.call_args.kwargs["research_report"]
    assert saved_report.answer == "Draft retained for follow-up questions."
    assert saved_report.confidence_score == pytest.approx(0.45)
    redis_client.close.assert_called_once()


def test_rejected_draft_routes_second_question_to_fast_chat():
    from app.workflows.rag_workflow import (
        _save_research_context,
        run_chat_workflow,
    )

    server = fakeredis.FakeServer()

    def make_redis_client():
        return fakeredis.FakeRedis(server=server)

    fast_llm = MagicMock()
    fast_llm.invoke.return_value.content = "RAG retrieves context before answering."
    initial_result = {
        "reviewed_answer": None,
        "draft_answer": "A rejected but usable RAG research draft.",
        "confidence_score": 0.45,
        "review_feedback": "Add inline citations.",
        "external_context": [],
    }

    with (
        patch("app.workflows.rag_workflow.create_redis_client", side_effect=make_redis_client),
        patch("app.workflows.rag_workflow.get_safe_llm", return_value=fast_llm),
    ):
        _save_research_context("session-1", initial_result)
        result = run_chat_workflow(
            question="Explain RAG briefly.",
            session_id="session-1",
        )

    assert result["answer"] == "RAG retrieves context before answering."
    assert result["is_fast_chat"] is True
    assert result["citations"] == []
    fast_llm.invoke.assert_called_once()


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
    assert MODEL_CANDIDATES["fast_chat"][0] == "openai/gpt-oss-20b:free"
