from unittest.mock import MagicMock, patch

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


def test_reviewer_uses_best_draft_when_all_providers_fail():
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

    assert result.reviewed_answer == state.draft_answer
    assert result.iteration_count == 1
    assert result.confidence_score == pytest.approx(0.69)
    assert "reviewer was unavailable" in result.review_feedback.lower()


def test_removed_unavailable_glm_free_slug():
    from app.core.model_candidates import MODEL_CANDIDATES

    candidates = [model for models in MODEL_CANDIDATES.values() for model in models]
    assert "z-ai/glm-4.5-air:free" not in candidates
