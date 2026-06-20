"""
Property-based tests for Deep Research Optimization.
Feature: deep-research-optimization
Uses Hypothesis with @settings(max_examples=100).
"""

import pytest
from hypothesis import given, settings, strategies as st

# ── Property 9: Cache Idempotence and Normalization ───────────────────────────
# Feature: deep-research-optimization, Property 9: Cache Idempotence and Normalization


@given(st.text())
@settings(max_examples=100)
def test_normalize_query_idempotence(q: str) -> None:
    """
    normalize_query is idempotent: applying it twice yields the same result as
    applying it once.

    normalize_query(normalize_query(q)) == normalize_query(q) for all q.

    Validates: Requirements 5.5, 5.6
    """
    from app.core.llm_cache import normalize_query

    once = normalize_query(q)
    twice = normalize_query(once)
    assert once == twice, (
        f"normalize_query is not idempotent.\n"
        f"  Input:       {q!r}\n"
        f"  First pass:  {once!r}\n"
        f"  Second pass: {twice!r}"
    )


@given(
    # Restrict to printable ASCII (codepoints 32–126) so that str.upper() /
    # str.lower() are stable inverses of each other.  Non-ASCII letters can
    # have asymmetric case-folding (e.g. 'ı' → 'I' → 'i') which would make
    # the test assertion incorrect rather than revealing a bug in normalize_query.
    st.text(
        min_size=1,
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    ),
    st.lists(
        st.sampled_from([" ", "  ", "\t", "\n"]),
        min_size=0,
        max_size=3,
    ),
    st.lists(
        st.sampled_from(["?", "!", ".", ",", ";", ":"]),
        min_size=0,
        max_size=3,
    ),
)
@settings(max_examples=100)
def test_same_normalized_form_same_cache_key(
    base: str,
    extra_whitespace: list,
    trailing_punct: list,
) -> None:
    """
    Two queries that differ only in case, surrounding whitespace, or trailing
    punctuation must produce the same normalized form — and therefore the same
    cache key — so that PlannerLLMCache treats them as equivalent.

    The base string is restricted to ASCII letters/digits plus common
    punctuation so that Python's str.upper() / str.lower() are stable
    (no Unicode case-folding edge cases such as µ → Μ → μ).

    Validates: Requirements 5.5, 5.6
    """
    from app.core.llm_cache import normalize_query, _cache_key

    # Variant 1: uppercase + leading/trailing spaces
    variant1 = "  " + base.upper() + "  "

    # Variant 2: lowercase + trailing punctuation appended
    punct_suffix = "".join(trailing_punct)
    variant2 = base.lower() + punct_suffix

    norm1 = normalize_query(variant1)
    norm2 = normalize_query(variant2)

    # Both variants must normalize to the same string as the plain base
    norm_base = normalize_query(base)

    assert norm1 == norm_base, (
        f"Upper-cased / padded variant did not normalize to base.\n"
        f"  base:     {base!r}\n"
        f"  variant1: {variant1!r}\n"
        f"  norm1:    {norm1!r}\n"
        f"  expected: {norm_base!r}"
    )
    assert norm2 == norm_base, (
        f"Trailing-punctuation variant did not normalize to base.\n"
        f"  base:     {base!r}\n"
        f"  variant2: {variant2!r}\n"
        f"  norm2:    {norm2!r}\n"
        f"  expected: {norm_base!r}"
    )

    # Same normalized form → same cache key
    assert _cache_key(variant1) == _cache_key(base), (
        f"Cache keys differ for equivalent queries.\n"
        f"  variant1: {variant1!r}\n"
        f"  base:     {base!r}"
    )
    assert _cache_key(variant2) == _cache_key(base), (
        f"Cache keys differ for equivalent queries.\n"
        f"  variant2: {variant2!r}\n"
        f"  base:     {base!r}"
    )


# ── Property 5: Job Store Round-Trip ─────────────────────────────────────────
# Feature: deep-research-optimization, Property 5: Job Store Round-Trip


# Strategy: generate job data dicts with realistic keys and JSON-serialisable values
_job_value_strategy = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=50),
)

_job_data_strategy = st.fixed_dictionaries(
    {
        "status": st.sampled_from(["pending", "done", "error"]),
        "result": st.one_of(st.none(), st.text(max_size=100)),
        "error": st.one_of(st.none(), st.text(max_size=100)),
    },
    optional={
        "session_id": st.text(min_size=1, max_size=36),
        "confidence_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    },
)


@given(
    task_id=st.text(min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="-_")),
    data=_job_data_strategy,
)
@settings(max_examples=100, deadline=None)
def test_job_store_round_trip(task_id: str, data: dict) -> None:
    """
    For any job data dict, create_job(task_id, data) then get_job(task_id)
    returns an equivalent dict — all fields preserved with the same values.

    Uses RedisJobStore in fallback mode by patching redis.from_url to raise
    ConnectionError, so the in-memory dict is used instead.

    Validates: Requirements 2.7
    """
    from unittest.mock import patch
    from app.core.job_store import RedisJobStore

    with patch("redis.from_url", side_effect=ConnectionError("Redis unavailable")):
        store = RedisJobStore()

    # Confirm we are in fallback mode (no Redis)
    assert not store._use_redis, "Expected in-memory fallback mode"

    store.create_job(task_id, data)
    retrieved = store.get_job(task_id)

    assert retrieved is not None, (
        f"get_job returned None after create_job.\n"
        f"  task_id: {task_id!r}\n"
        f"  data:    {data!r}"
    )
    assert retrieved == data, (
        f"Round-trip data mismatch.\n"
        f"  task_id:   {task_id!r}\n"
        f"  stored:    {data!r}\n"
        f"  retrieved: {retrieved!r}"
    )


# ── Property 6: Fast Chat Single LLM Call with Bounded Context ───────────────
# Feature: deep-research-optimization, Property 6: Fast Chat Single LLM Call with Bounded Context


def _make_message(role: str, content: str):
    """Helper: build a minimal Message object for testing."""
    from app.schemas.chat_models import Message, MessageRole
    return Message(
        session_id="test-session",
        role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
        content=content,
    )


def _make_context_window(messages: list, report_answer: str = "Test research answer."):
    """Helper: build a ContextWindow with a ResearchReport and given messages."""
    from app.schemas.chat_models import ContextWindow, ResearchReport, Source
    report = ResearchReport(answer=report_answer)
    source = Source(index=1, type="url", title="Test Source", url="https://example.com")
    return ContextWindow(
        session_id="test-session",
        messages=messages,
        research_report=report,
        sources=[source],
    )


@given(
    messages=st.lists(
        st.builds(
            _make_message,
            role=st.sampled_from(["user", "assistant"]),
            # Restrict content to printable ASCII to avoid multi-line content
            # that would interfere with the history-text reconstruction check.
            content=st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            ),
        ),
        min_size=0,
        max_size=30,
    ),
    question=st.text(
        min_size=1,
        max_size=200,
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    ),
)
@settings(max_examples=100, deadline=None)
def test_fast_chat_single_llm_call_and_bounded_context(messages: list, question: str) -> None:
    """
    For any ContextWindow (with research_report present) and any follow-up
    question string, FastChatAgent.run() must:
      1. Invoke the LLM exactly once (Requirement 3.3).
      2. Pass at most 10 conversation history entries in the HumanMessage
         content, regardless of how many messages are in the session history
         (Requirement 3.4).

    Validates: Requirements 3.3, 3.4
    """
    from unittest.mock import MagicMock
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.agents.fast_chat import FastChatAgent

    # --- Set up mock LLM ---
    mock_response = MagicMock()
    mock_response.content = "Mock fast chat answer."

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    # --- Build context and run agent ---
    context = _make_context_window(messages)
    agent = FastChatAgent(llm=mock_llm)
    agent.run(question, context)

    # --- Property 1: LLM invoked exactly once (Requirement 3.3) ---
    assert mock_llm.invoke.call_count == 1, (
        f"FastChatAgent.run() must invoke LLM exactly once, "
        f"but invoke was called {mock_llm.invoke.call_count} time(s).\n"
        f"  history length: {len(messages)}"
    )

    # --- Property 2: At most 10 history entries in the HumanMessage (Requirement 3.4) ---
    # Retrieve the messages list passed to llm.invoke()
    call_args = mock_llm.invoke.call_args
    passed_messages = call_args[0][0]  # first positional argument

    # There must be exactly 2 messages: [SystemMessage, HumanMessage]
    assert len(passed_messages) == 2, (
        f"Expected 2 messages passed to LLM (SystemMessage + HumanMessage), "
        f"got {len(passed_messages)}."
    )
    assert isinstance(passed_messages[0], SystemMessage), (
        "First message passed to LLM must be a SystemMessage."
    )
    assert isinstance(passed_messages[1], HumanMessage), (
        "Second message passed to LLM must be a HumanMessage."
    )

    human_content: str = passed_messages[1].content

    # The agent slices context.messages[-10:] before building history_text.
    # We verify the bounded-context property by:
    #   (a) Confirming the slice itself has at most 10 entries.
    #   (b) Confirming the HumanMessage embeds exactly the history text built
    #       from that bounded slice — not from the full message list.
    #
    # This approach is robust to message content containing newlines or other
    # special characters, because we reconstruct the expected text from the
    # same slice the agent uses and check for substring containment.

    expected_slice = context.messages[-10:]

    # (a) Slice length must be ≤ 10
    assert len(expected_slice) <= 10, (
        f"context.messages[-10:] has {len(expected_slice)} entries — "
        f"this should never exceed 10.\n"
        f"  Total messages in session: {len(messages)}"
    )

    # (b) The HumanMessage must contain the history text built from the bounded slice
    expected_history_text = "\n".join(
        f"{m.role.upper()}: {m.content}"
        for m in expected_slice
    )
    assert expected_history_text in human_content, (
        f"HumanMessage does not contain the expected bounded history text.\n"
        f"  Expected slice size: {len(expected_slice)} (of {len(messages)} total)\n"
        f"  Expected history text: {expected_history_text!r}\n"
        f"  HumanMessage content (first 500 chars): {human_content[:500]!r}"
    )

    # (c) If there are more than 10 messages, verify the full history is NOT embedded
    #     (i.e., the agent truly truncated to the last 10, not the full list).
    if len(messages) > 10:
        full_history_text = "\n".join(
            f"{m.role.upper()}: {m.content}"
            for m in messages
        )
        assert full_history_text not in human_content, (
            f"HumanMessage contains the FULL history ({len(messages)} messages) "
            f"instead of the bounded last-10 slice.\n"
            f"  This means the agent is not enforcing the 10-message limit "
            f"(Requirement 3.4)."
        )


# ── Property 7: Fast Chat Response Format ────────────────────────────────────
# Feature: deep-research-optimization, Property 7: Fast Chat Response Format


@given(
    question=st.text(
        min_size=1,
        max_size=200,
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    ),
    messages=st.lists(
        st.builds(
            _make_message,
            role=st.sampled_from(["user", "assistant"]),
            content=st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            ),
        ),
        min_size=0,
        max_size=20,
    ),
)
@settings(max_examples=100, deadline=None)
def test_fast_chat_response_format(question: str, messages: list) -> None:
    """
    For any valid ContextWindow and follow-up question, FastChatAgent.run()
    must return a dict containing all required fields with the correct types:
      - answer:             non-empty str
      - citations:          list
      - confidence_score:   float in [0, 1]
      - need_clarification: bool
      - is_fast_chat:       True

    Validates: Requirements 3.7
    """
    from unittest.mock import MagicMock
    from app.agents.fast_chat import FastChatAgent

    # --- Set up mock LLM that returns a non-empty response ---
    mock_response = MagicMock()
    mock_response.content = "This is a non-empty fast chat answer."

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    # --- Build ContextWindow with a ResearchReport ---
    context = _make_context_window(messages)

    # --- Run the agent ---
    agent = FastChatAgent(llm=mock_llm)
    result = agent.run(question, context)

    # --- Assert all required keys are present ---
    required_keys = {"answer", "citations", "confidence_score", "need_clarification", "is_fast_chat"}
    missing_keys = required_keys - result.keys()
    assert not missing_keys, (
        f"FastChatAgent.run() result is missing required keys: {missing_keys}\n"
        f"  Returned keys: {set(result.keys())}"
    )

    # --- Assert 'answer' is a non-empty string ---
    assert isinstance(result["answer"], str), (
        f"'answer' must be a str, got {type(result['answer']).__name__!r}.\n"
        f"  Value: {result['answer']!r}"
    )
    assert len(result["answer"]) > 0, (
        f"'answer' must be a non-empty string, but got an empty string."
    )

    # --- Assert 'citations' is a list ---
    assert isinstance(result["citations"], list), (
        f"'citations' must be a list, got {type(result['citations']).__name__!r}.\n"
        f"  Value: {result['citations']!r}"
    )

    # --- Assert 'confidence_score' is a float in [0, 1] ---
    assert isinstance(result["confidence_score"], float), (
        f"'confidence_score' must be a float, got {type(result['confidence_score']).__name__!r}.\n"
        f"  Value: {result['confidence_score']!r}"
    )
    assert 0.0 <= result["confidence_score"] <= 1.0, (
        f"'confidence_score' must be in [0, 1], got {result['confidence_score']!r}."
    )

    # --- Assert 'need_clarification' is a bool ---
    assert isinstance(result["need_clarification"], bool), (
        f"'need_clarification' must be a bool, got {type(result['need_clarification']).__name__!r}.\n"
        f"  Value: {result['need_clarification']!r}"
    )

    # --- Assert 'is_fast_chat' is True ---
    assert result["is_fast_chat"] is True, (
        f"'is_fast_chat' must be True for FastChatAgent responses, "
        f"got {result['is_fast_chat']!r}."
    )


# ── Property 8: Fast Chat Word Count Limit ───────────────────────────────────
# Feature: deep-research-optimization, Property 8: Fast Chat Word Count Limit


@given(
    # Generate a list of words (each word is a non-empty text token up to 10 chars),
    # then join them with spaces to form an answer with at most 800 words.
    word_list=st.lists(
        st.text(min_size=1, max_size=10),
        min_size=1,
        max_size=800,
    ),
    question=st.text(
        min_size=1,
        max_size=200,
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    ),
    messages=st.lists(
        st.builds(
            _make_message,
            role=st.sampled_from(["user", "assistant"]),
            content=st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            ),
        ),
        min_size=0,
        max_size=20,
    ),
)
@settings(max_examples=100, deadline=None)
def test_fast_chat_word_count_limit(word_list: list, question: str, messages: list) -> None:
    """
    Property 8: Fast Chat Word Count Limit.

    For any fast chat response, the word count of the ``answer`` field must be
    at most 800 words (Requirements 4.2, 4.3).

    This property tests the PROMPT constraint, not the agent routing logic.
    The FastChatAgent passes the word count constraint to the LLM via
    FAST_CHAT_SYSTEM_PROMPT.  Since we use a mock LLM in tests, we verify:

      1. FAST_CHAT_SYSTEM_PROMPT contains the word count constraints "500" and
         "800" so the LLM receives the correct instructions.
      2. When the mock LLM returns a response whose word count is ≤ 800, the
         agent passes it through unchanged — i.e., the agent does not truncate
         or alter the answer.

    Validates: Requirements 4.2, 4.3
    """
    from unittest.mock import MagicMock
    from app.agents.fast_chat import FastChatAgent
    from app.prompts.fast_chat_prompt import FAST_CHAT_SYSTEM_PROMPT

    # --- Part 1: Verify FAST_CHAT_SYSTEM_PROMPT contains word count constraints ---
    assert "500" in FAST_CHAT_SYSTEM_PROMPT, (
        "FAST_CHAT_SYSTEM_PROMPT must contain '500' to enforce the 500-word limit "
        "for regular questions (Requirement 4.2)."
    )
    assert "800" in FAST_CHAT_SYSTEM_PROMPT, (
        "FAST_CHAT_SYSTEM_PROMPT must contain '800' to enforce the 800-word limit "
        "for comparison/analysis questions (Requirement 4.3)."
    )

    # --- Part 2: Agent passes through a compliant answer unchanged ---
    # Build an answer whose word count is guaranteed to be ≤ 800
    generated_answer = " ".join(word_list)
    assert len(generated_answer.split()) <= 800, (
        f"Test setup error: generated answer has {len(generated_answer.split())} words, "
        f"expected ≤ 800."
    )

    mock_response = MagicMock()
    mock_response.content = generated_answer

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    context = _make_context_window(messages)
    agent = FastChatAgent(llm=mock_llm)
    result = agent.run(question, context)

    # The agent must return the LLM's answer verbatim (no truncation by the agent)
    assert result["answer"] == generated_answer, (
        f"FastChatAgent.run() must return the LLM answer verbatim.\n"
        f"  Expected: {generated_answer!r}\n"
        f"  Got:      {result['answer']!r}"
    )

    # The returned answer must satisfy the ≤ 800 word constraint
    word_count = len(result["answer"].split())
    assert word_count <= 800, (
        f"answer word count {word_count} exceeds the 800-word limit "
        f"(Requirements 4.2, 4.3)."
    )


# ── Property 1: Parallel Search Correctness ──────────────────────────────────
# Feature: deep-research-optimization, Property 1: Parallel Search Correctness


@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
@settings(max_examples=100, deadline=None)
def test_parallel_search_correctness(queries: list) -> None:
    """
    For any non-empty query list, _collect_sources_parallel returns the same
    URL set as the sequential implementation (_collect_sources).

    Both functions are called with the same mock for academic_search that
    returns deterministic results per query, so order may differ but no
    results should be lost or duplicated.

    Validates: Requirements 1.2
    """
    from unittest.mock import patch
    from app.agents.researcher import _collect_sources_parallel, _collect_sources

    def mock_academic_search(q: str):
        # Return a deterministic single result per query keyed by the query string
        return [{"url": f"https://example.com/{q}", "title": q, "content": "test"}]

    with patch("app.agents.researcher.academic_search", side_effect=mock_academic_search):
        parallel_results = _collect_sources_parallel(queries)
        sequential_results = _collect_sources(queries)

    parallel_urls = {r["url"] for r in parallel_results}
    sequential_urls = {r["url"] for r in sequential_results}

    assert parallel_urls == sequential_urls, (
        f"Parallel and sequential search returned different URL sets.\n"
        f"  queries:          {queries!r}\n"
        f"  parallel URLs:    {sorted(parallel_urls)}\n"
        f"  sequential URLs:  {sorted(sequential_urls)}\n"
        f"  only in parallel: {sorted(parallel_urls - sequential_urls)}\n"
        f"  only in sequential: {sorted(sequential_urls - parallel_urls)}"
    )


# ── Property 2: Query Count Cap ──────────────────────────────────────────────
# Feature: deep-research-optimization, Property 2: Query Count Cap


@given(st.lists(st.text(min_size=1, max_size=50), min_size=6, max_size=20))
@settings(max_examples=100, deadline=None)
def test_query_count_cap(queries: list) -> None:
    """
    For any query list with length > MAX_QUERIES (5), at most MAX_QUERIES calls
    to academic_search are executed by _collect_sources_parallel.

    Validates: Requirements 1.4
    """
    from unittest.mock import patch, MagicMock
    from app.agents.researcher import _collect_sources_parallel, MAX_QUERIES

    mock_academic_search = MagicMock(return_value=[])

    with patch("app.agents.researcher.academic_search", mock_academic_search):
        _collect_sources_parallel(queries)

    assert mock_academic_search.call_count <= MAX_QUERIES, (
        f"academic_search was called {mock_academic_search.call_count} time(s), "
        f"but must be called at most MAX_QUERIES={MAX_QUERIES} times.\n"
        f"  queries length: {len(queries)}\n"
        f"  queries: {queries!r}"
    )


# Feature: deep-research-optimization, Property 3: Review Router Acceptance


@given(
    confidence_score=st.floats(
        min_value=0.7,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    iteration_count=st.integers(min_value=0, max_value=20),
    max_iterations=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100, deadline=None)
def test_review_router_accepts_high_confidence(
    confidence_score: float,
    iteration_count: int,
    max_iterations: int,
) -> None:
    """
    For any AgentState with confidence_score >= 0.7, _review_router returns
    END regardless of iteration_count or max_iterations.

    Validates: Requirements 1.5
    """
    from langgraph.graph import END
    from app.graph.build_graph import _review_router
    from app.workflows.states import AgentState

    state = AgentState(
        question="test",
        reviewed_answer=None,
        confidence_score=confidence_score,
        iteration_count=iteration_count,
        max_iterations=max_iterations,
    )

    assert _review_router(state) == END


# ── Property 4: Research Context Round-Trip ───────────────────────────────────
# Feature: deep-research-optimization, Property 4: Research Context Round-Trip


@given(
    reviewed_answer=st.text(
        min_size=1,
        max_size=200,
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    )
)
@settings(max_examples=100, deadline=None)
def test_research_context_round_trip(reviewed_answer: str) -> None:
    """
    Property 4: Research Context Round-Trip.

    For any pipeline result with non-empty ``reviewed_answer``, storing the
    Research_Context via ``MemoryStore.init_session_context()`` and then
    retrieving it via ``MemoryStore.get_context_window()`` must return a
    ``ContextWindow`` whose ``research_report.answer`` equals the original
    answer.

    Uses a ``MagicMock`` to simulate Redis get/set/delete with an in-memory
    dict, so no real Redis connection is required.

    Validates: Requirements 1.7
    """
    import json
    from unittest.mock import MagicMock
    from app.core.memory_store import MemoryStore
    from app.schemas.chat_models import ResearchReport

    session_id = "test-session-round-trip"

    # ── Build an in-memory Redis simulation ──────────────────────────────────
    # MemoryStore uses hset / hgetall / expire on a Hash key.
    # We simulate this with a plain dict: _store[key][field] = value.
    _store: dict = {}

    mock_redis = MagicMock()

    def fake_hset(key, mapping=None, **kwargs):
        if key not in _store:
            _store[key] = {}
        if mapping:
            _store[key].update(mapping)

    def fake_hgetall(key):
        # Return a copy so mutations don't affect the store
        return dict(_store.get(key, {}))

    def fake_hget(key, field):
        return _store.get(key, {}).get(field)

    def fake_expire(key, ttl):
        pass  # TTL not needed for in-memory simulation

    def fake_exists(key):
        return key in _store

    mock_redis.hset.side_effect = fake_hset
    mock_redis.hgetall.side_effect = fake_hgetall
    mock_redis.hget.side_effect = fake_hget
    mock_redis.expire.side_effect = fake_expire
    mock_redis.exists.side_effect = fake_exists

    # ── Build the ResearchReport and store it ────────────────────────────────
    research_report = ResearchReport(answer=reviewed_answer)
    store = MemoryStore(mock_redis)

    store.init_session_context(
        session_id=session_id,
        research_report=research_report,
        sources=[],
    )

    # ── Retrieve and assert round-trip correctness ───────────────────────────
    context = store.get_context_window(session_id)

    assert context.research_report is not None, (
        f"get_context_window() returned a ContextWindow with research_report=None.\n"
        f"  session_id:      {session_id!r}\n"
        f"  reviewed_answer: {reviewed_answer!r}"
    )

    assert context.research_report.answer == reviewed_answer, (
        f"Round-trip answer mismatch.\n"
        f"  session_id: {session_id!r}\n"
        f"  stored:     {reviewed_answer!r}\n"
        f"  retrieved:  {context.research_report.answer!r}"
    )
