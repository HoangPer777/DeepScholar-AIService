"""
Bug Condition Exploration Test — Upload Embedding Chunks Fix

**Validates: Requirements 1.3**

Goal: Surface counterexamples proving that `process_pdf_pipeline` returns
{"status": "success"} even when `ingest_article_chunks` returns
{"stored": False, "error": "..."}.

EXPECTED OUTCOME: This test FAILS on unfixed code — failure confirms the bug exists.
"""

import pytest
from unittest.mock import patch, MagicMock
from hypothesis import given, settings as h_settings, HealthCheck
from hypothesis import strategies as st

from app.api.pdf import process_pdf_pipeline, PDFUploadRequest


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_bug_condition(pipeline_result: dict, ingest_result: dict) -> bool:
    """
    FUNCTION isBugCondition(pipeline_result, ingest_result)
      RETURN ingest_result.get("stored") == False
             AND ingest_result.get("error") IS NOT None
             AND pipeline_result.get("status") == "success"
             AND "embedding_error" NOT IN pipeline_result
    END FUNCTION
    """
    return (
        ingest_result.get("stored") == False
        and ingest_result.get("error") is not None
        and pipeline_result.get("status") == "success"
        and "embedding_error" not in pipeline_result
    )


# ── Strategies ────────────────────────────────────────────────────────────────

article_id_strategy = st.integers(min_value=1, max_value=10_000)

# Non-empty error strings
error_string_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=200,
)

chunk_strategy = st.lists(
    st.text(min_size=1, max_size=100),
    min_size=1,
    max_size=5,
)


# ── Fixtures / shared mock setup ──────────────────────────────────────────────

def make_request(article_id: int) -> PDFUploadRequest:
    return PDFUploadRequest(
        pdf_url="https://example.com/test.pdf",
        slug=f"test-article-{article_id}",
        article_id=article_id,
    )


def run_pipeline_with_failed_ingest(article_id: int, error: str) -> tuple[dict, dict]:
    """
    Run process_pdf_pipeline with all external deps mocked.
    ingest_article_chunks is mocked to return {"stored": False, "error": error}.
    Returns (pipeline_result, ingest_result).
    """
    ingest_result = {"stored": False, "error": error}

    fake_pdf_bytes = b"%PDF-1.4 fake content"
    fake_extracted = {
        "title": "Test Paper Title",
        "abstract": "Test abstract.",
        "content": "Test content body. " * 10,
    }
    fake_chunks = ["chunk one", "chunk two"]

    with patch("app.api.pdf._download_pdf_from_r2", return_value=fake_pdf_bytes), \
         patch("app.api.pdf.extract_sections_with_llamaparse", return_value=fake_extracted), \
         patch("app.api.pdf.chunk_text", return_value=fake_chunks), \
         patch("app.api.pdf.requests.patch") as mock_patch, \
         patch("app.api.pdf.ingest_article_chunks", return_value=ingest_result):

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_patch.return_value = mock_response

        request = make_request(article_id)
        pipeline_result = process_pdf_pipeline(request)

    return pipeline_result, ingest_result


# ── Property 1: Bug Condition — Embedding Failure Silently Ignored ────────────

@given(
    article_id=article_id_strategy,
    error=error_string_strategy,
)
@h_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property1_bug_condition_embedding_failure_reflected(article_id, error):
    """
    **Validates: Requirements 1.3**

    Property 1: Bug Condition — Embedding Failure Silently Ignored

    For any article_id (int) and non-empty error string, when ingest_article_chunks
    returns {"stored": False, "error": <error>}, process_pdf_pipeline MUST NOT
    return {"status": "success"} without an "embedding_error" key.

    Assert: result.get("status") != "success" OR "embedding_error" in result

    EXPECTED OUTCOME ON UNFIXED CODE: FAILS
    Counterexample: process_pdf_pipeline returns {"status": "success", "title": ..., ...}
    with no "embedding_error" key, even though ingest_article_chunks returned stored=False.
    """
    pipeline_result, ingest_result = run_pipeline_with_failed_ingest(article_id, error)

    # This assertion FAILS on unfixed code — proving the bug exists.
    # On fixed code it will PASS.
    assert (
        pipeline_result.get("status") != "success"
        or "embedding_error" in pipeline_result
    ), (
        f"BUG DETECTED: process_pdf_pipeline returned status='success' "
        f"without 'embedding_error' key, even though ingest_article_chunks "
        f"returned {ingest_result!r}. "
        f"Pipeline result: {pipeline_result!r}"
    )


# ── Property 2: Preservation — Successful Pipeline Behavior Unchanged ─────────

"""
**Validates: Requirements 3.1, 3.2, 3.3**

Observation-first methodology:
- Observed on UNFIXED code: when ingest_article_chunks returns {"stored": True, "chunk_count": N},
  process_pdf_pipeline returns {"status": "success", "title": ..., "abstract": ..., "content": ...}
- Observed on UNFIXED code: when sync=False, upload_pdf returns
  {"status": "processing_started", "slug": ..., "message": ...} immediately
- Observed on UNFIXED code: when sync=True and pipeline succeeds, upload_pdf returns
  {"status": "completed", "slug": ..., "data": ...}

EXPECTED OUTCOME: Tests PASS on unfixed code (confirms baseline behavior to preserve).
"""

from app.api.pdf import upload_pdf
from fastapi import BackgroundTasks

# ── Strategies for preservation tests ────────────────────────────────────────

slug_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="-",
    ),
    min_size=1,
    max_size=50,
)

non_empty_chunk_list_strategy = st.lists(
    st.text(min_size=1, max_size=100),
    min_size=1,
    max_size=5,
)


def run_pipeline_with_successful_ingest(article_id: int, chunks: list) -> dict:
    """
    Run process_pdf_pipeline with all external deps mocked.
    ingest_article_chunks is mocked to return {"stored": True, "chunk_count": len(chunks)}.
    Returns pipeline_result.
    """
    ingest_result = {"stored": True, "chunk_count": len(chunks)}

    fake_pdf_bytes = b"%PDF-1.4 fake content"
    fake_extracted = {
        "title": "Test Paper Title",
        "abstract": "Test abstract.",
        "content": "Test content body. " * 10,
    }

    with patch("app.api.pdf._download_pdf_from_r2", return_value=fake_pdf_bytes), \
         patch("app.api.pdf.extract_sections_with_llamaparse", return_value=fake_extracted), \
         patch("app.api.pdf.chunk_text", return_value=chunks), \
         patch("app.api.pdf.requests.patch") as mock_patch, \
         patch("app.api.pdf.ingest_article_chunks", return_value=ingest_result):

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_patch.return_value = mock_response

        request = make_request(article_id)
        return process_pdf_pipeline(request)


@given(
    article_id=article_id_strategy,
    chunks=non_empty_chunk_list_strategy,
)
@h_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property2a_preservation_successful_pipeline_returns_success(article_id, chunks):
    """
    **Validates: Requirements 3.1**

    Property 2a: Preservation — Successful Pipeline Returns Success

    For all article_id (integers) and non-empty chunk lists, when ingest_article_chunks
    is mocked to return {"stored": True, "chunk_count": len(chunks)}, process_pdf_pipeline
    MUST return {"status": "success"} with "title", "abstract", and "content" keys.

    EXPECTED OUTCOME: PASSES on unfixed code (confirms baseline behavior to preserve).
    """
    result = run_pipeline_with_successful_ingest(article_id, chunks)

    assert result.get("status") == "success", (
        f"Expected status='success' when ingest succeeds, got: {result!r}"
    )
    assert "title" in result, f"Expected 'title' key in result, got: {result!r}"
    assert "abstract" in result, f"Expected 'abstract' key in result, got: {result!r}"
    assert "content" in result, f"Expected 'content' key in result, got: {result!r}"


@given(slug=slug_strategy)
@h_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property2b_preservation_async_upload_returns_processing_started(slug):
    """
    **Validates: Requirements 3.2**

    Property 2b: Preservation — sync=False Returns Immediately

    For all slug strings, when upload_pdf is called with sync=False,
    it MUST return {"status": "processing_started"} immediately with "slug" and "message" keys,
    without waiting for the pipeline to complete.

    EXPECTED OUTCOME: PASSES on unfixed code (confirms baseline behavior to preserve).
    """
    fake_pdf_bytes = b"%PDF-1.4 fake content"
    fake_extracted = {
        "title": "Test Paper Title",
        "abstract": "Test abstract.",
        "content": "Test content body. " * 10,
    }
    ingest_result = {"stored": True, "chunk_count": 2}

    with patch("app.api.pdf._download_pdf_from_r2", return_value=fake_pdf_bytes), \
         patch("app.api.pdf.extract_sections_with_llamaparse", return_value=fake_extracted), \
         patch("app.api.pdf.chunk_text", return_value=["chunk one", "chunk two"]), \
         patch("app.api.pdf.requests.patch") as mock_patch, \
         patch("app.api.pdf.ingest_article_chunks", return_value=ingest_result):

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_patch.return_value = mock_response

        request = PDFUploadRequest(
            pdf_url="https://example.com/test.pdf",
            slug=slug if slug else "test-slug",
            article_id=1,
        )
        background_tasks = BackgroundTasks()
        result = upload_pdf(request, background_tasks, sync=False)

    assert result.get("status") == "processing_started", (
        f"Expected status='processing_started' for sync=False, got: {result!r}"
    )
    assert "slug" in result, f"Expected 'slug' key in result, got: {result!r}"
    assert "message" in result, f"Expected 'message' key in result, got: {result!r}"


@given(
    article_id=article_id_strategy,
    chunks=non_empty_chunk_list_strategy,
)
@h_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property2c_preservation_sync_upload_returns_completed(article_id, chunks):
    """
    **Validates: Requirements 3.3**

    Property 2c: Preservation — sync=True with Successful Pipeline Returns Completed

    For all article_id and chunk lists, when upload_pdf is called with sync=True
    and the pipeline succeeds (ingest returns stored=True), it MUST return
    {"status": "completed"} with "slug" and "data" keys.

    EXPECTED OUTCOME: PASSES on unfixed code (confirms baseline behavior to preserve).
    """
    ingest_result = {"stored": True, "chunk_count": len(chunks)}

    fake_pdf_bytes = b"%PDF-1.4 fake content"
    fake_extracted = {
        "title": "Test Paper Title",
        "abstract": "Test abstract.",
        "content": "Test content body. " * 10,
    }

    with patch("app.api.pdf._download_pdf_from_r2", return_value=fake_pdf_bytes), \
         patch("app.api.pdf.extract_sections_with_llamaparse", return_value=fake_extracted), \
         patch("app.api.pdf.chunk_text", return_value=chunks), \
         patch("app.api.pdf.requests.patch") as mock_patch, \
         patch("app.api.pdf.ingest_article_chunks", return_value=ingest_result):

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_patch.return_value = mock_response

        request = PDFUploadRequest(
            pdf_url="https://example.com/test.pdf",
            slug=f"test-article-{article_id}",
            article_id=article_id,
        )
        background_tasks = BackgroundTasks()
        result = upload_pdf(request, background_tasks, sync=True)

    assert result.get("status") == "completed", (
        f"Expected status='completed' for sync=True with successful pipeline, got: {result!r}"
    )
    assert "slug" in result, f"Expected 'slug' key in result, got: {result!r}"
    assert "data" in result, f"Expected 'data' key in result, got: {result!r}"
