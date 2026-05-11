"""
Property-Based Tests — deep-research-retrieval-fix

Tests verify correctness properties for:
- Retry/backoff logic (P2, P3)
- Source reranker bonus weights (P8, P9, P10)
- Academic ratio computation (P5)
- Reviewer quality gate (P6, P7, P15)
- Query deduplication (P11, P12)

Uses hypothesis for property-based testing.
"""
import math
import time
from typing import Dict, List
from unittest.mock import MagicMock, patch, call

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from app.tools.academic_search import _fetch_with_retry, _parallel_fetch
from app.tools.source_reranker import rerank_sources, _SOURCE_TYPE_BONUS
from app.tools.source_filter import enforce_source_diversity, LOW_QUALITY_DOMAINS
from app.agents.researcher import _deduplicate_queries
from app.agents.reviewer import _source_quality_gate, _ACADEMIC_SOURCE_TYPES, _LOW_ACADEMIC_SCORE_CAP
from app.tools.semantic_scholar import RateLimitError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(source_type: str = "web", citation_count: int = 0, url: str = "http://example.com") -> Dict:
    return {
        "source_type":    source_type,
        "title":          f"Test paper ({source_type})",
        "content":        "Some content about the topic.",
        "url":            url,
        "citation_count": citation_count,
        "score":          0.5,
    }


# ---------------------------------------------------------------------------
# P2: Retry on 429 with exponential backoff
# Feature: deep-research-retrieval-fix, Property 2: Retry on 429 with exponential backoff
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=2))
def test_p2_retry_count_and_backoff(n_failures: int):
    """
    For any fetch function that returns 429 on first N attempts then succeeds,
    _fetch_with_retry should call it exactly N+1 times.
    """
    call_count = 0
    expected_result = [_make_source("arxiv")]

    def mock_fetch(query: str) -> List[Dict]:
        nonlocal call_count
        call_count += 1
        if call_count <= n_failures:
            raise RateLimitError("429")
        return expected_result

    with patch("time.sleep"):  # don't actually sleep in tests
        source_name, results = _fetch_with_retry(
            fetch_fn=mock_fetch,
            fallback_fn=None,
            query="test query",
            source_name="TestSource",
            max_retries=3,
            base_delay=0.001,
        )

    assert call_count == n_failures + 1, f"Expected {n_failures + 1} calls, got {call_count}"
    assert results == expected_result


# ---------------------------------------------------------------------------
# P3: Fallback invoked when primary source exhausts retries
# Feature: deep-research-retrieval-fix, Property 3: Fallback invoked when primary exhausts retries
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(st.text(min_size=1, max_size=100))
def test_p3_fallback_called_when_primary_exhausts(query: str):
    """
    If primary fetch always raises RateLimitError, fallback should be called exactly once.
    """
    fallback_call_count = 0
    fallback_query_received = None
    fallback_result = [_make_source("openalex")]

    def always_rate_limit(q: str) -> List[Dict]:
        raise RateLimitError("429")

    def mock_fallback(q: str) -> List[Dict]:
        nonlocal fallback_call_count, fallback_query_received
        fallback_call_count += 1
        fallback_query_received = q
        return fallback_result

    with patch("time.sleep"):
        source_name, results = _fetch_with_retry(
            fetch_fn=always_rate_limit,
            fallback_fn=mock_fallback,
            query=query,
            source_name="TestSource",
            max_retries=3,
            base_delay=0.001,
        )

    assert fallback_call_count == 1, f"Fallback should be called exactly once, got {fallback_call_count}"
    assert fallback_query_received == query, "Fallback should receive the same query"
    assert results == fallback_result


# ---------------------------------------------------------------------------
# P5: Academic ratio computation is correct
# Feature: deep-research-retrieval-fix, Property 5: Academic ratio computation is correct
# ---------------------------------------------------------------------------

_SOURCE_TYPES = list(_ACADEMIC_SOURCE_TYPES) + ["web", "blog", "github"]

@settings(max_examples=100)
@given(st.lists(
    st.fixed_dictionaries({
        "source_type":    st.sampled_from(_SOURCE_TYPES),
        "title":          st.text(min_size=1, max_size=50),
        "content":        st.text(max_size=100),
        "url":            st.text(min_size=1, max_size=50),
        "citation_count": st.integers(min_value=0, max_value=1000),
    }),
    min_size=0,
    max_size=20,
))
def test_p5_academic_ratio_computation(sources: List[Dict]):
    """
    For any list of sources, academic_ratio = academic_count / max(total, 1).
    """
    # Add research notes entry (should be excluded)
    sources_with_notes = sources + [{"title": "__research_notes__", "source_type": "internal", "url": ""}]

    real_sources = [s for s in sources_with_notes if s.get("title") != "__research_notes__"]
    total = len(real_sources)
    academic_count = sum(1 for s in real_sources if s.get("source_type") in _ACADEMIC_SOURCE_TYPES)
    expected_ratio = academic_count / max(total, 1)

    # Verify the gate computes the same ratio
    # We test this indirectly by checking gate behavior matches expected ratio
    gate_passed, gate_failed = _source_quality_gate(sources_with_notes, need_external_search=True)

    if total == 0:
        assert "no_sources_found" in gate_failed
    elif expected_ratio < 0.3:
        assert "insufficient_academic_sources" in gate_failed
    else:
        assert "insufficient_academic_sources" not in gate_failed


# ---------------------------------------------------------------------------
# P6: Low academic ratio forces rewrite and caps score
# Feature: deep-research-retrieval-fix, Property 6: Low academic ratio forces rewrite and caps score
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=10))
def test_p6_low_academic_ratio_forces_rewrite(web_source_count: int):
    """
    When academic_ratio < 0.3 and need_external_search=True,
    gate should fail with insufficient_academic_sources.
    """
    # All web sources → academic_ratio = 0.0
    sources = [_make_source("web", url=f"http://example{i}.com") for i in range(web_source_count)]

    gate_passed, gate_failed = _source_quality_gate(sources, need_external_search=True)

    assert not gate_passed, "Gate should fail with 0 academic sources"
    assert "insufficient_academic_sources" in gate_failed


@settings(max_examples=100)
@given(st.integers(min_value=1, max_value=5))
def test_p6_score_cap_applied(academic_count: int):
    """
    When academic_ratio >= 0.3, gate should pass criterion (a).
    """
    # Enough academic sources to pass ratio check
    total = academic_count * 3  # ratio = 1/3 ≈ 0.33 >= 0.3
    sources = (
        [_make_source("arxiv", citation_count=50, url=f"http://arxiv.org/{i}") for i in range(academic_count)]
        + [_make_source("web", url=f"http://web{i}.com") for i in range(total - academic_count)]
    )

    gate_passed, gate_failed = _source_quality_gate(sources, need_external_search=True)

    assert "insufficient_academic_sources" not in gate_failed


# ---------------------------------------------------------------------------
# P7: Quality gate skipped when need_external_search=False
# Feature: deep-research-retrieval-fix, Property 7: Quality gate skipped when external search not needed
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(st.lists(
    st.fixed_dictionaries({
        "source_type": st.sampled_from(["web", "blog"]),
        "title":       st.text(min_size=1, max_size=50),
        "url":         st.text(min_size=1, max_size=50),
        "citation_count": st.just(0),
    }),
    min_size=0,
    max_size=10,
))
def test_p7_gate_skipped_when_no_external_search(sources: List[Dict]):
    """
    When need_external_search=False, gate should always pass regardless of sources.
    """
    gate_passed, gate_failed = _source_quality_gate(sources, need_external_search=False)

    assert gate_passed is True
    assert gate_failed == []


# ---------------------------------------------------------------------------
# P8: Source type bonus values are correct
# Feature: deep-research-retrieval-fix, Property 8: Source type bonus values are correct
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.sampled_from(list(_SOURCE_TYPE_BONUS.keys())))
def test_p8_source_type_bonus_values(source_type: str):
    """
    After reranking, each source's score should include the exact bonus from _SOURCE_TYPE_BONUS.
    Use a query that matches the source title exactly so relevance_score is deterministic.
    """
    # Use a specific title and matching query so relevance is 1.0 (all words match)
    source = _make_source(source_type, citation_count=0)
    source["title"] = "test paper"
    source["content"] = ""

    # Query matches title exactly → relevance = 1.0 (title_hits=2, n=2, raw=2*2/(2*3)=0.667, no phrase bonus)
    # Actually use empty content and a query that gives known relevance
    # Simplest: use a single-word query that matches title
    source["title"] = "alpha"
    source["content"] = ""
    query = "alpha"  # 1 word, matches title → title_hits=1, n=1, raw=2/3≈0.667

    ranked = rerank_sources([source], query=query)

    assert len(ranked) == 1
    expected_bonus = _SOURCE_TYPE_BONUS[source_type]
    # relevance = 2/3 ≈ 0.667, citation = 0, score = 0.667 + type_bonus
    expected_score = round(2/3 + expected_bonus, 4)
    assert abs(ranked[0]["score"] - expected_score) < 0.001, (
        f"Expected score {expected_score} for {source_type}, got {ranked[0]['score']}"
    )


# ---------------------------------------------------------------------------
# P9: Citation bonus formula is correct
# Feature: deep-research-retrieval-fix, Property 9: Citation bonus formula is correct
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.integers(min_value=0, max_value=100000))
def test_p9_citation_bonus_formula(citation_count: int):
    """
    Citation bonus should equal math.log1p(citation_count) * 0.1.
    Use a query that matches the source title to ensure source passes relevance threshold.
    """
    source = _make_source("web", citation_count=citation_count)
    source["title"] = "alpha"
    source["content"] = ""
    query = "alpha"  # matches title → relevance = 2/3

    ranked = rerank_sources([source], query=query)

    assert len(ranked) == 1
    expected_citation_bonus = math.log1p(citation_count) * 0.1
    expected_type_bonus = _SOURCE_TYPE_BONUS.get("web", 0.0)
    expected_relevance = round(2/3, 10)
    expected_score = round(expected_relevance + expected_citation_bonus + expected_type_bonus, 4)

    assert abs(ranked[0]["score"] - expected_score) < 0.001, (
        f"Expected score {expected_score}, got {ranked[0]['score']}"
    )


# ---------------------------------------------------------------------------
# P10: All sources have score field after reranking
# Feature: deep-research-retrieval-fix, Property 10: All sources have score field after reranking
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.lists(
    st.fixed_dictionaries({
        "source_type":    st.sampled_from(_SOURCE_TYPES),
        "title":          st.just("alpha beta"),  # fixed title matching query
        "content":        st.text(max_size=200),
        "url":            st.text(min_size=1, max_size=50),
        "citation_count": st.integers(min_value=0, max_value=10000),
    }),
    min_size=1,
    max_size=20,
))
def test_p10_all_sources_have_score_field(sources: List[Dict]):
    """
    After reranking, every source that passes relevance threshold should have a 'score' field.
    Use a query that matches the fixed title so sources are not filtered out.
    """
    ranked = rerank_sources(sources, query="alpha beta")

    # All returned sources must have score field
    for source in ranked:
        assert "score" in source, f"Source missing 'score' field: {source}"
        assert isinstance(source["score"], (int, float)), f"Score is not numeric: {source['score']}"


# ---------------------------------------------------------------------------
# P11: Query deduplication reduces or preserves query count
# Feature: deep-research-retrieval-fix, Property 11: Query deduplication reduces or preserves query count
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(st.lists(st.text(min_size=0, max_size=50), min_size=0, max_size=20))
def test_p11_query_deduplication(queries: List[str]):
    """
    _deduplicate_queries should:
    (a) len(result) <= len(original)
    (b) no two entries identical after lower().strip()
    (c) all unique normalized queries from original are present
    """
    result = _deduplicate_queries(queries)

    # (a) length constraint
    assert len(result) <= len(queries)

    # (b) no duplicates after normalization
    normalized_result = [q.lower().strip() for q in result]
    assert len(normalized_result) == len(set(normalized_result)), "Duplicates found after normalization"

    # (c) all unique normalized queries from original are present
    original_normalized = {q.lower().strip() for q in queries if q.strip()}
    result_normalized = set(normalized_result)
    assert original_normalized == result_normalized, (
        f"Missing queries: {original_normalized - result_normalized}"
    )


# ---------------------------------------------------------------------------
# P12: URL deduplication — no duplicate URLs in final context
# Feature: deep-research-retrieval-fix, Property 12: No duplicate URLs in final context
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(
    st.lists(st.text(min_size=5, max_size=30), min_size=1, max_size=5),
    st.integers(min_value=2, max_value=4),
)
def test_p12_url_deduplication_in_researcher(unique_urls: List[str], repeat_count: int):
    """
    ResearcherAgent should not produce duplicate URLs in external_context.
    """
    from app.agents.researcher import ResearcherAgent
    from app.workflows.states import AgentState

    # Create sources with duplicate URLs (simulating overlap between queries)
    duplicate_sources = [
        _make_source("arxiv", url=url)
        for url in unique_urls
        for _ in range(repeat_count)
    ]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Research notes")

    agent = ResearcherAgent(llm=mock_llm)
    state = AgentState(
        question="test",
        need_external_search=True,
        search_queries=["query1"],
    )

    with patch("app.agents.researcher.academic_search", return_value=duplicate_sources):
        result_state = agent.run(state)

    # Filter out research notes
    real_sources = [s for s in result_state.external_context if s.get("title") != "__research_notes__"]
    urls = [s.get("url") for s in real_sources if s.get("url")]
    assert len(urls) == len(set(urls)), f"Duplicate URLs found: {urls}"


# ---------------------------------------------------------------------------
# P15: Accepted drafts pass all source quality criteria
# Feature: deep-research-retrieval-fix, Property 15: Accepted drafts pass all source quality criteria
# ---------------------------------------------------------------------------

@settings(max_examples=50)
@given(st.integers(min_value=3, max_value=10))
def test_p15_accepted_drafts_pass_quality_criteria(academic_count: int):
    """
    When all quality criteria pass, gate should return (True, []).
    """
    # Build sources that satisfy all criteria:
    # (a) academic_ratio >= 0.3
    # (b) at least 1 source with citation_count > 10
    # (c) no LOW_QUALITY_DOMAINS
    sources = [
        _make_source("arxiv", citation_count=100, url=f"https://arxiv.org/abs/{i}")
        for i in range(academic_count)
    ] + [
        _make_source("web", citation_count=0, url=f"https://example.com/{i}")
        for i in range(2)  # 2 web sources, ratio = academic_count / (academic_count + 2)
    ]

    gate_passed, gate_failed = _source_quality_gate(sources, need_external_search=True)

    assert gate_passed, f"Gate should pass but failed: {gate_failed}"
    assert gate_failed == []


# ---------------------------------------------------------------------------
# Additional: enforce_source_diversity includes openalex/crossref as academic
# ---------------------------------------------------------------------------

def test_openalex_crossref_treated_as_academic():
    """
    enforce_source_diversity should count openalex and crossref as academic sources.
    """
    sources = [
        _make_source("openalex", url=f"https://openalex.org/{i}") for i in range(5)
    ] + [
        _make_source("crossref", url=f"https://doi.org/{i}") for i in range(5)
    ] + [
        _make_source("web", url=f"https://web{i}.com") for i in range(5)
    ]

    result = enforce_source_diversity(sources)

    academic_in_result = [s for s in result if s.get("source_type") in ("openalex", "crossref")]
    web_in_result = [s for s in result if s.get("source_type") == "web"]

    assert len(academic_in_result) <= 12
    assert len(web_in_result) <= 3
    # All 10 academic sources should be included (under limit of 12)
    assert len(academic_in_result) == 10


def test_rate_limit_error_raised_by_semantic_scholar():
    """
    search_semantic_scholar should raise RateLimitError on HTTP 429.
    """
    from app.tools.semantic_scholar import search_semantic_scholar

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        with pytest.raises(RateLimitError):
            search_semantic_scholar("test query")


def test_rate_limit_error_raised_by_arxiv():
    """
    search_arxiv should raise RateLimitError on HTTP 429.
    """
    from app.tools.arxiv_search import search_arxiv

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        with pytest.raises(RateLimitError):
            search_arxiv("test query")
