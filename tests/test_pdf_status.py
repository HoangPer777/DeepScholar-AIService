import os
from unittest.mock import MagicMock

os.environ.setdefault("INTERNAL_SERVICE_KEY", "test-internal-key")

from fastapi.testclient import TestClient

from app.api import pdf
from app.main import app


class FakeJobStore:
    def __init__(self):
        self.jobs = {}
        self.is_persistent = True

    def create_job(self, task_id, data):
        self.jobs[task_id] = data

    def get_job(self, task_id):
        return self.jobs.get(task_id)

    def update_job(self, task_id, data):
        self.jobs[task_id] = data

    def create_job_if_absent(self, task_id, data):
        if task_id in self.jobs:
            return False
        self.jobs[task_id] = data
        return True

    def get_pdf_job(self, task_id):
        job = self.jobs.get(task_id)
        return job if job and job.get('kind') == 'pdf' else None

    def update_pdf_job(self, task_id, updates):
        job = self.get_pdf_job(task_id)
        if not job or job.get('status') in {'completed', 'failed'}:
            return job
        job.update(updates)
        return job


def test_async_upload_returns_task_id_and_queues_job(monkeypatch):
    store = FakeJobStore()
    monkeypatch.setattr(pdf, "_job_store", store)
    request = pdf.PDFUploadRequest(
        task_id="20000000-0000-0000-0000-000000000001",
        generation=1,
        content_revision=0,
        idempotency_key="20000000-0000-0000-0000-000000000002",
        pdf_url="https://example.com/paper.pdf",
        slug="status-paper",
        article_id=42,
    )
    background_tasks = MagicMock()

    response = pdf.upload_pdf(request, background_tasks, sync=False, x_internal_service_key="test-internal-key")

    assert response["status"] == "processing_started"
    assert response["task_id"] in store.jobs
    assert response["progress"]["phase"] == "queued"
    background_tasks.add_task.assert_called_once()


def test_status_returns_progress_for_valid_internal_request(monkeypatch):
    store = FakeJobStore()
    store.jobs["task-1"] = {
        "kind": "pdf",
        "task_id": "task-1",
        "status": "running",
        "slug": "status-paper",
        "article_id": 42,
        "generation": 1,
        "content_revision": 0,
        "progress": {
            "phase": "embedding",
            "message": "Creating and storing document embeddings…",
            "percent": 85,
        },
    }
    monkeypatch.setattr(pdf, "_job_store", store)
    client = TestClient(app)

    response = client.get(
        "/api/pdf/status",
        params={"task_id": "task-1"},
        headers={"X-Internal-Service-Key": "test-internal-key"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["progress"]["phase"] == "embedding"
    assert response.json()["article_id"] == 42
    assert response.json()["generation"] == 1


def test_status_rejects_missing_internal_key(monkeypatch):
    store = FakeJobStore()
    monkeypatch.setattr(pdf, "_job_store", store)
    client = TestClient(app)

    response = client.get("/api/pdf/status", params={"task_id": "task-1"})

    assert response.status_code == 403


def test_pdf_upload_rejects_missing_internal_key():
    client = TestClient(app)
    response = client.post(
        "/api/pdf/upload",
        json={
            "task_id": "20000000-0000-0000-0000-000000000007",
            "generation": 1,
            "content_revision": 0,
            "idempotency_key": "20000000-0000-0000-0000-000000000008",
            "pdf_url": "https://example.com/paper.pdf",
            "slug": "status-paper",
            "article_id": 42,
        },
    )

    assert response.status_code == 403


def test_status_returns_404_for_expired_task(monkeypatch):
    store = FakeJobStore()
    monkeypatch.setattr(pdf, "_job_store", store)
    client = TestClient(app)

    response = client.get(
        "/api/pdf/status",
        params={"task_id": "expired-task"},
        headers={"X-Internal-Service-Key": "test-internal-key"},
    )

    assert response.status_code == 404


def test_status_does_not_expose_other_job_kinds(monkeypatch):
    store = FakeJobStore()
    store.jobs["chat-task"] = {"kind": "chat", "status": "done", "result": {"secret": "hidden"}}
    monkeypatch.setattr(pdf, "_job_store", store)
    client = TestClient(app)

    response = client.get(
        "/api/pdf/status",
        params={"task_id": "chat-task"},
        headers={"X-Internal-Service-Key": "test-internal-key"},
    )

    assert response.status_code == 404
    assert "hidden" not in response.text


def test_status_returns_503_when_shared_store_is_unavailable(monkeypatch):
    class UnavailableStore(FakeJobStore):
        def get_pdf_job(self, _task_id):
            raise RuntimeError("redis down")

    monkeypatch.setattr(pdf, "_job_store", UnavailableStore())
    client = TestClient(app)

    response = client.get(
        "/api/pdf/status",
        params={"task_id": "pdf-task"},
        headers={"X-Internal-Service-Key": "test-internal-key"},
    )

    assert response.status_code == 503
    assert "redis down" not in response.text


def test_pdf_job_transitions_to_completed(monkeypatch):
    store = FakeJobStore()
    store.jobs["task-success"] = {
        "kind": "pdf",
        "task_id": "task-success",
        "status": "queued",
        "slug": "status-paper",
        "article_id": 42,
        "progress": {"phase": "queued", "message": "Queued", "percent": 0},
    }
    monkeypatch.setattr(pdf, "_job_store", store)

    def fake_pipeline(_request, progress_callback=None):
        progress_callback({"phase": "extracting", "message": "Extracting", "percent": 35})
        return {"status": "success"}

    monkeypatch.setattr(pdf, "process_pdf_pipeline", fake_pipeline)
    request = pdf.PDFUploadRequest(
        task_id="20000000-0000-0000-0000-000000000003",
        generation=1,
        content_revision=0,
        idempotency_key="20000000-0000-0000-0000-000000000004",
        pdf_url="https://example.com/paper.pdf",
        slug="status-paper",
        article_id=42,
    )

    pdf._run_pdf_job("task-success", request)

    assert store.jobs["task-success"]["status"] == "completed"
    assert store.jobs["task-success"]["progress"]["phase"] == "completed"


def test_pdf_job_transitions_to_failed_without_exposing_exception(monkeypatch):
    store = FakeJobStore()
    store.jobs["task-failure"] = {
        "kind": "pdf",
        "task_id": "task-failure",
        "status": "queued",
        "slug": "status-paper",
        "article_id": 42,
        "progress": {"phase": "queued", "message": "Queued", "percent": 0},
    }
    monkeypatch.setattr(pdf, "_job_store", store)

    def fake_pipeline(_request, progress_callback=None):
        return {
            "status": "error",
            "error_code": "article_persist_failed",
            "error": "The extracted article could not be saved.",
        }

    monkeypatch.setattr(pdf, "process_pdf_pipeline", fake_pipeline)
    request = pdf.PDFUploadRequest(
        task_id="20000000-0000-0000-0000-000000000005",
        generation=1,
        content_revision=0,
        idempotency_key="20000000-0000-0000-0000-000000000006",
        pdf_url="https://example.com/paper.pdf",
        slug="status-paper",
        article_id=42,
    )

    pdf._run_pdf_job("task-failure", request)

    job = store.jobs["task-failure"]
    assert job["status"] == "failed"
    assert job["error_code"] == "article_persist_failed"
    assert job["article_saved"] is False
    assert "https://example.com" not in job["message"]
