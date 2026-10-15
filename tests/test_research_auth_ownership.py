from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import research
from app.core.auth import AuthenticatedUser, get_current_user
from app.main import app


class FakeJobStore:
    def __init__(self):
        self.jobs = {}

    def create_job(self, task_id, data):
        self.jobs[task_id] = dict(data)

    def get_job(self, task_id):
        return self.jobs.get(task_id)

    def update_job(self, task_id, data):
        self.jobs[task_id] = dict(data)


@pytest.fixture(autouse=True)
def clean_overrides():
    previous = app.dependency_overrides.copy()
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


def _set_identity(user_id):
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id)


def test_research_start_persists_owner_from_verified_principal(monkeypatch):
    store = FakeJobStore()
    monkeypatch.setattr(research, "_job_store", store)
    created_sessions = []
    monkeypatch.setattr(
        research,
        "_create_research_session",
        lambda session_id, owner_id, query: created_sessions.append((session_id, owner_id, query)),
    )
    monkeypatch.setattr(research.asyncio, "create_task", lambda coro: coro.close())
    _set_identity("user-12")

    response = TestClient(app).post(
        "/api/research/deep-research",
        json={"query": "test research ownership", "user_id": "forged-user"},
    )

    assert response.status_code == 200
    task_id = response.json()["task_id"]
    assert store.jobs[task_id]["kind"] == "research"
    assert store.jobs[task_id]["owner_id"] == "user-12"
    assert store.jobs[task_id]["session_id"] == task_id
    assert created_sessions == [(task_id, "user-12", "test research ownership")]


def test_research_status_isolated_and_terminal_poll_is_repeatable(monkeypatch):
    store = FakeJobStore()
    task_id = str(uuid4())
    store.jobs[task_id] = {
        "kind": "research",
        "owner_id": "alice",
        "session_id": str(uuid4()),
        "status": "done",
        "result": {"answer": "private research result"},
    }
    monkeypatch.setattr(research, "_job_store", store)
    monkeypatch.setattr(research, "_require_research_session_owner", lambda _session_id, _owner_id: None)
    client = TestClient(app)

    _set_identity("bob")
    denied = client.get(f"/api/research/status/{task_id}")
    assert denied.status_code == 404
    assert "private research result" not in denied.text

    _set_identity("alice")
    first = client.get(f"/api/research/status/{task_id}")
    second = client.get(f"/api/research/status/{task_id}")
    assert first.status_code == second.status_code == 200
    assert first.json()["answer"] == second.json()["answer"] == "private research result"


def test_status_rejects_legacy_jobs_without_owner(monkeypatch):
    store = FakeJobStore()
    task_id = str(uuid4())
    store.jobs[task_id] = {"status": "done", "result": {"answer": "legacy private"}}
    monkeypatch.setattr(research, "_job_store", store)
    monkeypatch.setattr(research, "_require_research_session_owner", lambda _session_id, _owner_id: None)
    _set_identity("alice")

    response = TestClient(app).get(f"/api/research/status/{task_id}")

    assert response.status_code == 404
    assert "legacy private" not in response.text
