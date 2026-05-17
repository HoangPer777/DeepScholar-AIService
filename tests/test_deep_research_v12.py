"""
Tests cho Deep Research V12 Upgrade.

Bao gồm:
- Unit tests cho từng component mới
- Property-Based Tests (PBT) với hypothesis cho các invariants

Chạy: python -m pytest tests/test_deep_research_v12.py -v
"""
import math
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.messages import AIMessage

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / Strategies
# ─────────────────────────────────────────────────────────────────────────────

def _make_source(source_type: str = "web", url: str = "https://example.com") -> Dict:
    return {
        "source_type":    source_type,
        "title":          "Test Paper",
        "content":        "Some content about research.",
        "url":            url,
        "citation_count": 0,
        "score":          0.5,
    }


@st.composite
def source_strategy(draw):
    """Hypothesis strategy for generating source dicts."""
    source_type = draw(st.sampled_from(["arxiv", "semantic_scholar", "web", "blog", "github"]))
    url = draw(st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="/:.-_"),
        min_size=5, max_size=50,
    ))
    citation_count = draw(st.integers(min_value=0, max_value=10000))
    return {
        "source_type":    source_type,
        "title":          draw(st.text(min_size=1, max_size=100)),
        "content":        draw(st.text(min_size=0, max_size=500)),
        "url":            f"https://{url}",
        "citation_count": citation_count,
        "score":          0.5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 5.1.3 — PBT: enforce_source_diversity invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceDiversity:
    """Tests for source_filter.enforce_source_diversity()"""

    def test_empty_input_returns_empty(self):
        from app.tools.source_filter import enforce_source_diversity
        assert enforce_source_diversity([]) == []

    def test_academic_sources_limited_to_12(self):
        from app.tools.source_filter import enforce_source_diversity
        sources = [_make_source("arxiv", f"https://arxiv.org/{i}") for i in range(20)]
        result = enforce_source_diversity(sources)
        academic = [s for s in result if s["source_type"] in ("arxiv", "semantic_scholar", "alphaxiv")]
        assert len(academic) <= 12

    def test_web_sources_limited_to_3(self):
        from app.tools.source_filter import enforce_source_diversity
        sources = [_make_source("web", f"https://example.com/{i}") for i in range(10)]
        result = enforce_source_diversity(sources)
        web = [s for s in result if s["source_type"] not in ("arxiv", "semantic_scholar", "alphaxiv")]
        assert len(web) <= 3

    def test_mixed_sources_respects_both_limits(self):
        from app.tools.source_filter import enforce_source_diversity
        sources = (
            [_make_source("arxiv", f"https://arxiv.org/{i}") for i in range(15)] +
            [_make_source("web", f"https://example.com/{i}") for i in range(8)]
        )
        result = enforce_source_diversity(sources)
        academic = [s for s in result if s["source_type"] in ("arxiv", "semantic_scholar", "alphaxiv")]
        web = [s for s in result if s["source_type"] not in ("arxiv", "semantic_scholar", "alphaxiv")]
        assert len(academic) <= 12
        assert len(web) <= 3

    @given(sources=st.lists(source_strategy(), max_size=50))
    @settings(max_examples=100)
    def test_pbt_diversity_invariant(self, sources):
        """
        PBT Property 3: ∀ source lists → academic<=12, web<=3
        """
        from app.tools.source_filter import enforce_source_diversity
        result = enforce_source_diversity(sources)
        academic = [s for s in result if s.get("source_type") in ("arxiv", "semantic_scholar", "alphaxiv")]
        web = [s for s in result if s.get("source_type") not in ("arxiv", "semantic_scholar", "alphaxiv")]
        assert len(academic) <= 12, f"Academic count {len(academic)} exceeds 12"
        assert len(web) <= 3, f"Web count {len(web)} exceeds 3"


# ─────────────────────────────────────────────────────────────────────────────
# Low-quality filter tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceFilter:
    """Tests for source_filter.filter_low_quality_sources()"""

    def test_removes_reddit(self):
        from app.tools.source_filter import filter_low_quality_sources
        sources = [_make_source("web", "https://reddit.com/r/ml")]
        assert filter_low_quality_sources(sources) == []

    def test_removes_quora(self):
        from app.tools.source_filter import filter_low_quality_sources
        sources = [_make_source("web", "https://quora.com/what-is-rag")]
        assert filter_low_quality_sources(sources) == []

    def test_removes_linkedin(self):
        from app.tools.source_filter import filter_low_quality_sources
        sources = [_make_source("web", "https://linkedin.com/posts/123")]
        assert filter_low_quality_sources(sources) == []

    def test_removes_twitter_x(self):
        from app.tools.source_filter import filter_low_quality_sources
        sources = [
            _make_source("web", "https://twitter.com/user/status/123"),
            _make_source("web", "https://x.com/user/status/456"),
        ]
        assert filter_low_quality_sources(sources) == []

    def test_keeps_arxiv(self):
        from app.tools.source_filter import filter_low_quality_sources
        sources = [_make_source("arxiv", "https://arxiv.org/abs/2401.12345")]
        assert len(filter_low_quality_sources(sources)) == 1

    def test_keeps_semantic_scholar(self):
        from app.tools.source_filter import filter_low_quality_sources
        sources = [_make_source("semantic_scholar", "https://semanticscholar.org/paper/abc")]
        assert len(filter_low_quality_sources(sources)) == 1

    def test_empty_input(self):
        from app.tools.source_filter import filter_low_quality_sources
        assert filter_low_quality_sources([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Source reranker tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceReranker:
    """Tests for source_reranker.rerank_sources()"""

    def test_empty_input_returns_empty(self):
        from app.tools.source_reranker import rerank_sources
        assert rerank_sources([], "query") == []

    def test_returns_sorted_by_score(self):
        from app.tools.source_reranker import rerank_sources
        sources = [
            {**_make_source("web"), "citation_count": 0, "content": "unrelated content"},
            {**_make_source("arxiv"), "citation_count": 500, "content": "transformer attention mechanism"},
        ]
        result = rerank_sources(sources, "transformer attention")
        # arxiv with matching content + citations should rank higher
        assert result[0]["source_type"] == "arxiv"

    def test_arxiv_gets_type_bonus(self):
        from app.tools.source_reranker import rerank_sources
        sources = [
            {**_make_source("web", "https://blog.com"), "citation_count": 0, "content": "same content"},
            {**_make_source("arxiv", "https://arxiv.org/abs/1"), "citation_count": 0, "content": "same content"},
        ]
        result = rerank_sources(sources, "same content")
        assert result[0]["source_type"] == "arxiv"

    def test_citation_bonus_applied(self):
        from app.tools.source_reranker import rerank_sources
        # Use a query that matches the default title "Test Paper" so sources
        # pass the relevance threshold and are not dropped before ranking.
        sources = [
            {**_make_source("web", "https://a.com"), "citation_count": 0},
            {**_make_source("web", "https://b.com"), "citation_count": 1000},
        ]
        result = rerank_sources(sources, "test paper")
        # Higher citation count should rank higher (both web, same type bonus)
        assert result[0]["citation_count"] == 1000

    def test_score_field_added_to_sources(self):
        from app.tools.source_reranker import rerank_sources
        sources = [_make_source("arxiv")]
        result = rerank_sources(sources, "test query")
        assert "score" in result[0]
        assert isinstance(result[0]["score"], float)

    def test_empty_query_no_crash(self):
        from app.tools.source_reranker import rerank_sources
        sources = [_make_source("arxiv")]
        result = rerank_sources(sources, "")
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Task 5.1.4 — PBT: arXiv content minimum length
# ─────────────────────────────────────────────────────────────────────────────

class TestArxivSearch:
    """Tests for arxiv_search.search_arxiv()"""

    def test_returns_list_on_api_error(self):
        """search_arxiv() không raise exception khi API fail."""
        from app.tools.arxiv_search import search_arxiv
        with patch("app.tools.arxiv_search.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")
            result = search_arxiv("transformer")
            assert isinstance(result, list)
            assert result == []

    def test_returns_list_on_bad_status(self):
        from app.tools.arxiv_search import search_arxiv
        with patch("app.tools.arxiv_search.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 503
            mock_get.return_value = mock_resp
            result = search_arxiv("transformer")
            assert isinstance(result, list)

    def test_empty_query_returns_empty(self):
        from app.tools.arxiv_search import search_arxiv
        result = search_arxiv("")
        assert result == []

    def test_arxiv_entry_has_alphaxiv_url(self):
        """Mỗi arXiv entry phải có alphaxiv_url."""
        from app.tools.arxiv_search import search_arxiv

        # Mock feedparser response
        mock_entry = MagicMock()
        mock_entry.link = "https://arxiv.org/abs/2401.12345"
        mock_entry.title = "Test Paper"
        mock_entry.summary = "A" * 200  # 200 chars abstract
        mock_entry.authors = []
        mock_entry.published = "2024-01-01"

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<feed/>"

        with patch("app.tools.arxiv_search.requests.get", return_value=mock_resp), \
             patch("app.tools.arxiv_search.feedparser.parse", return_value=mock_feed):
            result = search_arxiv("test query")

        assert len(result) == 1
        assert result[0]["alphaxiv_url"] == "https://www.alphaxiv.org/abs/2401.12345"
        assert result[0]["url"] == "https://arxiv.org/abs/2401.12345"

    def test_arxiv_content_is_full_abstract(self):
        """content phải là entry.summary[:2000], không bị cắt ngắn hơn."""
        from app.tools.arxiv_search import search_arxiv

        long_summary = "X" * 1800  # 1800 chars — dưới 2000 nên không bị cắt

        mock_entry = MagicMock()
        mock_entry.link = "https://arxiv.org/abs/2401.99999"
        mock_entry.title = "Long Abstract Paper"
        mock_entry.summary = long_summary
        mock_entry.authors = []
        mock_entry.published = "2024-01-01"

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<feed/>"

        with patch("app.tools.arxiv_search.requests.get", return_value=mock_resp), \
             patch("app.tools.arxiv_search.feedparser.parse", return_value=mock_feed):
            result = search_arxiv("test")

        assert len(result) == 1
        assert len(result[0]["content"]) == 1800  # Không bị cắt

    @given(summary=st.text(min_size=100, max_size=3000))
    @settings(max_examples=50)
    def test_pbt_arxiv_content_min_length(self, summary):
        """
        PBT Property 4: ∀ summary với len >= 100 → content >= 100 chars
        """
        from app.tools.arxiv_search import search_arxiv

        mock_entry = MagicMock()
        mock_entry.link = "https://arxiv.org/abs/2401.00001"
        mock_entry.title = "Test"
        mock_entry.summary = summary
        mock_entry.authors = []
        mock_entry.published = "2024-01-01"

        mock_feed = MagicMock()
        mock_feed.entries = [mock_entry]

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<feed/>"

        with patch("app.tools.arxiv_search.requests.get", return_value=mock_resp), \
             patch("app.tools.arxiv_search.feedparser.parse", return_value=mock_feed):
            result = search_arxiv("test")

        if result:
            assert len(result[0]["content"]) >= 100, \
                f"Content too short: {len(result[0]['content'])} chars"


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Scholar tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSemanticScholar:
    """Tests for semantic_scholar.search_semantic_scholar()"""

    def test_returns_list_on_api_error(self):
        from app.tools.semantic_scholar import search_semantic_scholar
        with patch("app.tools.semantic_scholar.requests.get") as mock_get:
            mock_get.side_effect = Exception("Timeout")
            result = search_semantic_scholar("rag retrieval")
            assert isinstance(result, list)
            assert result == []

    def test_returns_list_on_bad_status(self):
        from app.tools.semantic_scholar import search_semantic_scholar, RateLimitError
        with patch("app.tools.semantic_scholar.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_get.return_value = mock_resp
            # 429 raises RateLimitError — caller (academic_search) handles it
            # The test verifies the exception type is correct
            with pytest.raises(RateLimitError):
                search_semantic_scholar("rag")

    def test_empty_query_returns_empty(self):
        from app.tools.semantic_scholar import search_semantic_scholar
        result = search_semantic_scholar("")
        assert result == []

    def test_parses_paper_fields(self):
        from app.tools.semantic_scholar import search_semantic_scholar

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{
                "title": "Attention Is All You Need",
                "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
                "year": 2017,
                "citationCount": 50000,
                "url": "https://semanticscholar.org/paper/abc123",
                "venue": "NeurIPS",
            }]
        }

        with patch("app.tools.semantic_scholar.requests.get", return_value=mock_resp):
            result = search_semantic_scholar("attention transformer")

        assert len(result) == 1
        paper = result[0]
        assert paper["source_type"] == "semantic_scholar"
        assert paper["title"] == "Attention Is All You Need"
        assert paper["citation_count"] == 50000
        assert paper["year"] == 2017
        assert "Ashish Vaswani" in paper["authors"]
        assert len(paper["content"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Task 5.1.2 — PBT: academic_search always returns list
# ─────────────────────────────────────────────────────────────────────────────

class TestAcademicSearch:
    """Tests for academic_search.academic_search()"""

    def test_returns_list_when_all_apis_fail(self):
        """academic_search() không raise exception kể cả khi tất cả APIs fail."""
        from app.tools.academic_search import academic_search
        with patch("app.tools.academic_search.search_semantic_scholar", side_effect=Exception("fail")), \
             patch("app.tools.academic_search.search_openalex", side_effect=Exception("fail")), \
             patch("app.tools.academic_search.tavily_search", side_effect=Exception("fail")):
            result = academic_search("transformer")
            assert isinstance(result, list)

    def test_combines_all_sources(self):
        from app.tools.academic_search import academic_search

        ss_result = [{**_make_source("semantic_scholar", "https://ss.org/1"), "citation_count": 10}]
        oa_result = [{**_make_source("openalex", "https://openalex.org/W1"), "citation_count": 0}]
        tv_result = [_make_source("web", "https://blog.com/1")]

        with patch("app.tools.academic_search.search_semantic_scholar", return_value=ss_result), \
             patch("app.tools.academic_search.search_openalex", return_value=oa_result), \
             patch("app.tools.academic_search.tavily_search", return_value=tv_result):
            result = academic_search("test query")

        assert isinstance(result, list)
        assert len(result) >= 1  # At least some results

    def test_deduplicates_by_url(self):
        from app.tools.academic_search import academic_search

        duplicate_url = "https://openalex.org/W2401"
        ss_result = [{**_make_source("semantic_scholar", duplicate_url), "citation_count": 5}]
        oa_result = [{**_make_source("openalex", duplicate_url), "citation_count": 0}]

        with patch("app.tools.academic_search.search_semantic_scholar", return_value=ss_result), \
             patch("app.tools.academic_search.search_openalex", return_value=oa_result), \
             patch("app.tools.academic_search.tavily_search", return_value=[]):
            result = academic_search("test")

        urls = [s["url"] for s in result]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"

    def test_enforces_diversity(self):
        from app.tools.academic_search import academic_search

        many_openalex = [{**_make_source("openalex", f"https://openalex.org/{i}"), "citation_count": 0} for i in range(20)]
        many_web = [_make_source("web", f"https://blog.com/{i}") for i in range(10)]

        with patch("app.tools.academic_search.search_semantic_scholar", return_value=[]), \
             patch("app.tools.academic_search.search_openalex", return_value=many_openalex), \
             patch("app.tools.academic_search.tavily_search", return_value=many_web):
            result = academic_search("test")

        academic = [s for s in result if s["source_type"] in ("arxiv", "semantic_scholar", "alphaxiv", "openalex", "crossref")]
        web = [s for s in result if s["source_type"] not in ("arxiv", "semantic_scholar", "alphaxiv", "openalex", "crossref")]
        assert len(academic) <= 12
        assert len(web) <= 3

    @given(query=st.text(max_size=200))
    @settings(max_examples=30)
    def test_pbt_always_returns_list(self, query):
        """
        PBT Property 2: ∀ query ∈ String → academic_search() returns list, no exception
        """
        from app.tools.academic_search import academic_search
        with patch("app.tools.academic_search.search_semantic_scholar", return_value=[]), \
             patch("app.tools.academic_search.search_openalex", return_value=[]), \
             patch("app.tools.academic_search.tavily_search", return_value=[]):
            result = academic_search(query)
            assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# Task 5.1.1 — PBT: SafeLLM never raises
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeLLM:
    """Tests for safe_llm.SafeLLM"""

    def _make_mock_groq(self, response_content: str = "mock response") -> MagicMock:
        mock = MagicMock()
        mock.invoke.return_value = AIMessage(content=response_content)
        return mock

    def test_uses_groq_when_no_openrouter_key(self):
        """Khi không có OPENROUTER_API_KEY → dùng Groq ngay."""
        from app.core.safe_llm import SafeLLM

        groq_mock = self._make_mock_groq("groq response")
        llm = SafeLLM("planner", ["openai/gpt-oss-120b:free"], groq_mock)

        with patch("app.core.safe_llm.settings") as mock_settings:
            mock_settings.OPENROUTER_API_KEY = ""
            result = llm.invoke([{"role": "user", "content": "test"}])

        groq_mock.invoke.assert_called_once()
        assert result.content == "groq response"

    def test_returns_first_successful_model(self):
        """Trả về kết quả từ model đầu tiên thành công."""
        from app.core.safe_llm import SafeLLM

        groq_mock = self._make_mock_groq()
        llm = SafeLLM("planner", ["model-a", "model-b"], groq_mock)

        mock_response = AIMessage(content="model-a response")

        with patch("app.core.safe_llm.settings") as mock_settings, \
             patch.object(llm, "_build_openrouter_llm") as mock_build:
            mock_settings.OPENROUTER_API_KEY = "test-key"
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_build.return_value = mock_llm

            result = llm.invoke([{"role": "user", "content": "test"}])

        assert result.content == "model-a response"
        groq_mock.invoke.assert_not_called()

    def test_falls_back_to_next_model_on_failure(self):
        """Khi model đầu fail → thử model tiếp theo."""
        from app.core.safe_llm import SafeLLM

        groq_mock = self._make_mock_groq()
        llm = SafeLLM("planner", ["model-a", "model-b"], groq_mock)

        call_count = 0
        def side_effect_build(model_name):
            mock_llm = MagicMock()
            if model_name == "model-a":
                mock_llm.invoke.side_effect = Exception("Rate limit")
            else:
                mock_llm.invoke.return_value = AIMessage(content="model-b response")
            return mock_llm

        with patch("app.core.safe_llm.settings") as mock_settings, \
             patch.object(llm, "_build_openrouter_llm", side_effect=side_effect_build), \
             patch("app.core.safe_llm.time.sleep"):
            mock_settings.OPENROUTER_API_KEY = "test-key"
            result = llm.invoke([{"role": "user", "content": "test"}])

        assert result.content == "model-b response"
        groq_mock.invoke.assert_not_called()

    def test_falls_back_to_groq_when_all_fail(self):
        """Khi tất cả OpenRouter models fail → dùng Groq emergency fallback."""
        from app.core.safe_llm import SafeLLM

        groq_mock = self._make_mock_groq("groq emergency response")
        llm = SafeLLM("planner", ["model-a", "model-b"], groq_mock)

        def always_fail(model_name):
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("All models down")
            return mock_llm

        with patch("app.core.safe_llm.settings") as mock_settings, \
             patch.object(llm, "_build_openrouter_llm", side_effect=always_fail), \
             patch("app.core.safe_llm.time.sleep"):
            mock_settings.OPENROUTER_API_KEY = "test-key"
            result = llm.invoke([{"role": "user", "content": "test"}])

        groq_mock.invoke.assert_called_once()
        assert result.content == "groq emergency response"

    @given(
        num_failing=st.integers(min_value=0, max_value=5),
        query=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=50)
    def test_pbt_never_raises(self, num_failing, query):
        """
        PBT Property 1: ∀ failure patterns → SafeLLM.invoke() không raise exception
        """
        from app.core.safe_llm import SafeLLM

        groq_mock = MagicMock()
        groq_mock.invoke.return_value = AIMessage(content="fallback")

        candidates = [f"model-{i}" for i in range(num_failing + 1)]
        llm = SafeLLM("planner", candidates, groq_mock)

        fail_count = [0]

        def build_with_failures(model_name):
            mock_llm = MagicMock()
            if fail_count[0] < num_failing:
                mock_llm.invoke.side_effect = Exception("Simulated failure")
                fail_count[0] += 1
            else:
                mock_llm.invoke.return_value = AIMessage(content="success")
            return mock_llm

        with patch("app.core.safe_llm.settings") as mock_settings, \
             patch.object(llm, "_build_openrouter_llm", side_effect=build_with_failures), \
             patch("app.core.safe_llm.time.sleep"):  # Skip sleep in tests
            mock_settings.OPENROUTER_API_KEY = "test-key"
            # Should never raise
            result = llm.invoke([{"role": "user", "content": query}])
            assert hasattr(result, "content")


# ─────────────────────────────────────────────────────────────────────────────
# Task 5.1.5 — Example: Researcher produces research_notes
# ─────────────────────────────────────────────────────────────────────────────

class TestResearcherAgent:
    """Tests for ResearcherAgent with academic_search integration."""

    def test_produces_research_notes_in_external_context(self):
        """
        ResearcherAgent.run() → external_context[0]["title"] == "__research_notes__"
        """
        from app.agents.researcher import ResearcherAgent
        from app.workflows.states import AgentState

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Research synthesis notes")

        mock_sources = [
            {**_make_source("arxiv", "https://arxiv.org/abs/1"), "citation_count": 0, "alphaxiv_url": "https://www.alphaxiv.org/abs/1"},
            {**_make_source("semantic_scholar", "https://ss.org/1"), "citation_count": 100},
        ]

        agent = ResearcherAgent(mock_llm)
        state = AgentState(
            question="What is RAG?",
            need_external_search=True,
            search_queries=["RAG retrieval augmented generation"],
        )

        with patch("app.agents.researcher.academic_search", return_value=mock_sources):
            result = agent.run(state)

        assert len(result.external_context) > 0
        assert result.external_context[0]["title"] == "__research_notes__"
        assert result.external_context[0]["content"] == "Research synthesis notes"

    def test_skips_when_no_external_search_needed(self):
        """ResearcherAgent skips khi need_external_search=False."""
        from app.agents.researcher import ResearcherAgent
        from app.workflows.states import AgentState

        mock_llm = MagicMock()
        agent = ResearcherAgent(mock_llm)
        state = AgentState(
            question="test",
            need_external_search=False,
        )

        with patch("app.agents.researcher.academic_search") as mock_search:
            result = agent.run(state)
            mock_search.assert_not_called()

        assert result.external_context == []

    def test_deduplicates_across_queries(self):
        """Không có duplicate URLs khi nhiều queries trả về cùng source."""
        from app.agents.researcher import ResearcherAgent
        from app.workflows.states import AgentState

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="notes")

        duplicate_source = {**_make_source("arxiv", "https://arxiv.org/abs/same"), "citation_count": 0}

        agent = ResearcherAgent(mock_llm)
        state = AgentState(
            question="test",
            need_external_search=True,
            search_queries=["query 1", "query 2"],
        )

        with patch("app.agents.researcher.academic_search", return_value=[duplicate_source]):
            result = agent.run(state)

        # Exclude __research_notes__
        real_sources = [s for s in result.external_context if s.get("title") != "__research_notes__"]
        urls = [s["url"] for s in real_sources]
        assert len(urls) == len(set(urls)), "Duplicate URLs in external_context"


# ─────────────────────────────────────────────────────────────────────────────
# Model candidates tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCandidates:
    """Tests for model_candidates.MODEL_CANDIDATES"""

    def test_all_agents_have_candidates(self):
        from app.core.model_candidates import MODEL_CANDIDATES
        for agent in ("planner", "clarifier", "researcher", "writer", "reviewer"):
            assert agent in MODEL_CANDIDATES, f"Missing candidates for {agent}"
            assert len(MODEL_CANDIDATES[agent]) > 0, f"Empty candidates for {agent}"

    def test_candidates_are_strings(self):
        from app.core.model_candidates import MODEL_CANDIDATES
        for agent, candidates in MODEL_CANDIDATES.items():
            for model in candidates:
                assert isinstance(model, str), f"Non-string model in {agent}: {model}"
                assert len(model) > 0, f"Empty model name in {agent}"


# ─────────────────────────────────────────────────────────────────────────────
# get_safe_llm tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSafeLlm:
    """Tests for llm.get_safe_llm()"""

    def test_returns_safe_llm_when_openrouter_key_set(self):
        from app.core.llm import get_safe_llm
        from app.core.safe_llm import SafeLLM

        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.OPENROUTER_API_KEY = "test-key"
            mock_settings.GROQ_API_KEY = "groq-key"
            mock_settings.GROQ_LLM_MODEL = "llama-3.1-8b-instant"
            result = get_safe_llm("planner")

        assert isinstance(result, SafeLLM)
        assert result.agent_name == "planner"

    def test_returns_groq_when_no_openrouter_key(self):
        from app.core.llm import get_safe_llm
        from app.core.safe_llm import SafeLLM

        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.OPENROUTER_API_KEY = ""
            mock_settings.GROQ_API_KEY = "groq-key"
            mock_settings.GROQ_LLM_MODEL = "llama-3.1-8b-instant"
            mock_settings.AGENT_LLM_PROVIDER = "groq"
            result = get_safe_llm("planner")

        # Should NOT be SafeLLM — falls back to get_agent_llm()
        assert not isinstance(result, SafeLLM)

    def test_get_agent_llm_still_works(self):
        """get_agent_llm() vẫn hoạt động sau V12 update."""
        from app.core.llm import get_agent_llm

        with patch("app.core.llm.settings") as mock_settings:
            mock_settings.AGENT_LLM_PROVIDER = "groq"
            mock_settings.GROQ_LLM_MODEL = "llama-3.1-8b-instant"
            mock_settings.GROQ_API_KEY = "test-key"
            result = get_agent_llm()

        assert result is not None
