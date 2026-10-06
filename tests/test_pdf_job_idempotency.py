from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import fakeredis
import pytest
from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from app.api import pdf
from app.core.job_store import RedisJobStore
from app.embeddings import vector_store


class FakePdfJobStore:
    is_persistent = True

    def __init__(self):
        self.jobs = {}

    def create_job_if_absent(self, task_id, data):
        if task_id in self.jobs:
            return False
        self.jobs[task_id] = data
        return True

    def get_job(self, task_id):
        return self.jobs.get(task_id)


def _request():
    return pdf.PDFUploadRequest(
        task_id="60000000-0000-0000-0000-000000000001",
        generation=4,
        content_revision=2,
        idempotency_key="60000000-0000-0000-0000-000000000002",
        pdf_url="https://example.test/paper.pdf",
        slug="paper",
        article_id=7,
    )


def test_redis_job_claim_is_atomic_for_duplicate_pdf_requests():
    store = RedisJobStore.__new__(RedisJobStore)
    store._fallback = {}
    store._fallback_lock = __import__("threading").Lock()
    store._use_redis = True
    store._redis = fakeredis.FakeRedis(decode_responses=True)
    payload = {"task_id": "same-task", "status": "queued"}

    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: store.create_job_if_absent("same-task", payload), range(32)))

    assert sum(claims) == 1
    assert store.get_job("same-task") == payload


def test_pdf_job_claim_is_idempotent_and_does_not_reset_progress(monkeypatch):
    store = FakePdfJobStore()
    monkeypatch.setattr(pdf, "_job_store", store)
    request = _request()

    assert pdf._create_pdf_job(request.task_id, request) is True
    store.jobs[request.task_id]["status"] = "running"
    store.jobs[request.task_id]["progress"] = {"phase": "embedding", "percent": 85}

    assert pdf._create_pdf_job(request.task_id, request) is False
    assert store.jobs[request.task_id]["status"] == "running"
    assert store.jobs[request.task_id]["progress"]["phase"] == "embedding"


def test_pdf_job_claim_rejects_task_identity_mismatch(monkeypatch):
    store = FakePdfJobStore()
    monkeypatch.setattr(pdf, "_job_store", store)
    request = _request()
    pdf._create_pdf_job(request.task_id, request)

    changed = request.model_copy(update={"generation": request.generation + 1})
    with pytest.raises(HTTPException) as error:
        pdf._create_pdf_job(request.task_id, changed)

    assert error.value.status_code == 409


def test_embedding_retry_reuses_same_generation_and_task(monkeypatch):
    request = _request()
    calls = []
    outcomes = [
        {"stored": False, "retryable": True, "error": "temporary database timeout"},
        {"stored": False, "retryable": True, "error": "temporary database timeout"},
        {"stored": True, "chunk_count": 1},
    ]

    def ingest(article_id, chunks, **kwargs):
        calls.append((article_id, kwargs.copy()))
        return outcomes.pop(0)

    monkeypatch.setattr(pdf.time, "sleep", lambda _seconds: None)
    result = pdf._ingest_pdf_chunks(request, ["chunk"], ingest)

    assert result["stored"] is True
    assert len(calls) == 3
    assert all(args[0] == request.article_id for args in calls)
    assert all(args[1] == {"pdf_generation": 4, "content_revision": 2} for args in calls)


def test_pdf_admission_fails_closed_when_redis_is_not_persistent(monkeypatch):
    store = FakePdfJobStore()
    store.is_persistent = False
    monkeypatch.setattr(pdf, "_job_store", store)

    with pytest.raises(HTTPException) as error:
        pdf.upload_pdf(
            _request(),
            BackgroundTasks(),
            sync=False,
            x_internal_service_key="test-internal-key",
        )

    assert error.value.status_code == 503
    assert store.jobs == {}


def test_pdf_similarity_query_fences_chunks_to_current_article_revision():
    query = (
        select(vector_store.ArticleChunk.id)
        .select_from(vector_store.ArticleChunk)
        .join(
            vector_store._articles,
            vector_store.ArticleChunk.article_id == vector_store._articles.c.id,
        )
        .where(vector_store._current_pdf_chunk_filter())
    )
    compiled_query = query.compile(dialect=postgresql.dialect())
    compiled = str(compiled_query)

    assert "pdf_generation" in compiled
    assert "content_revision" in compiled
    assert "__deepscholar_pdf_generation" in compiled_query.params.values()
    assert "__deepscholar_content_revision" in compiled_query.params.values()


def test_internal_pdf_fence_metadata_is_not_returned_to_clients():
    chunk = SimpleNamespace(
        id=3,
        content="current article content",
        section="results",
        section_title="Results",
        chunk_type="section_text",
        heading_path="[]",
        page_start=None,
        page_end=None,
        chunk_index=0,
        metadata_json={
            "source": "pdf",
            "__deepscholar_pdf_generation": 4,
            "__deepscholar_content_revision": 2,
        },
    )
    row = SimpleNamespace(ArticleChunk=chunk, distance=0.1)

    result = vector_store._row_to_result(row)

    assert result["metadata"] == {"source": "pdf"}
