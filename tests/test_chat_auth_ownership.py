from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import chatbot
from app.core import auth
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


class FakeAuthResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {"id": 12, "is_active": True}
        self.is_success = 200 <= status_code < 300

    def json(self):
        return self._body


@pytest.fixture(autouse=True)
def clean_overrides():
    previous = app.dependency_overrides.copy()
    yield
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)


def _set_identity(user_id):
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(user_id)


def test_missing_bearer_is_rejected_before_chat_or_history_access(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            calls.append(True)
            return FakeAuthResponse()

    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeClient)
    client = TestClient(app)

    response = client.get(f"/api/chat/history/{uuid4()}")

    assert response.status_code == 401
    assert calls == []


@pytest.mark.parametrize("backend_status,expected", [(401, 401), (403, 401), (500, 503)])
def test_backend_auth_failure_is_fail_closed(monkeypatch, backend_status, expected):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url, headers):
            assert headers == {"Authorization": "Bearer test-token"}
            return FakeAuthResponse(status_code=backend_status)

    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeClient)
    response = TestClient(app).get(
        f"/api/chat/history/{uuid4()}",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == expected


def test_auth_timeout_returns_503_without_anonymous_fallback(monkeypatch):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, *_args, **_kwargs):
            raise httpx.TimeoutException("test timeout")

    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeClient)
    response = TestClient(app).post("/api/chat/", json={"question": "test question"})

    assert response.status_code == 401

    response = TestClient(app).post(
        "/api/chat/",
        json={"question": "test question"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 503


def test_backend_identity_wins_over_client_supplied_user_id(monkeypatch):
    store = FakeJobStore()
    monkeypatch.setattr(chatbot, "_job_store", store)
    created_sessions = []
    monkeypatch.setattr(chatbot, "_create_owned_session", lambda sid, owner, query: created_sessions.append((sid, owner, query)))
    monkeypatch.setattr(chatbot.asyncio, "create_task", lambda coro: coro.close())

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, headers):
            assert url.endswith("/auth/me/")
            assert headers["Authorization"] == "Bearer signed-user-token"
            return FakeAuthResponse(body={"id": 12, "is_active": True})

    monkeypatch.setattr(auth.httpx, "AsyncClient", FakeClient)
    response = TestClient(app).post(
        "/api/chat/start",
        headers={"Authorization": "Bearer signed-user-token"},
        json={"question": "test authenticated job", "user_id": "999"},
    )

    assert response.status_code == 200
    task_id = response.json()["task_id"]
    assert store.jobs[task_id]["owner_id"] == "12"
    assert store.jobs[task_id]["kind"] == "chat"
    assert created_sessions[0][1] == "12"


def test_chat_task_results_are_owner_scoped_and_repeatable(monkeypatch):
    store = FakeJobStore()
    task_id = str(uuid4())
    store.jobs[task_id] = {
        "kind": "chat",
        "owner_id": "alice",
        "session_id": str(uuid4()),
        "status": "done",
        "result": {"answer": "private answer"},
    }
    monkeypatch.setattr(chatbot, "_job_store", store)
    monkeypatch.setattr(chatbot, "_require_owned_session", lambda _session_id, _owner_id: None)
    client = TestClient(app)

    _set_identity("bob")
    denied = client.get(f"/api/chat/status/{task_id}")
    assert denied.status_code == 404
    assert "private answer" not in denied.text

    _set_identity("alice")
    first = client.get(f"/api/chat/status/{task_id}")
    second = client.get(f"/api/chat/status/{task_id}")
    assert first.status_code == second.status_code == 200
    assert first.json()["answer"] == second.json()["answer"] == "private answer"


def test_history_checks_database_owner_before_reading_redis(monkeypatch):
    def deny_owner(_session_id, _owner_id):
        raise HTTPException(status_code=404, detail="Chat session not found.")

    redis_factory = Mock(side_effect=AssertionError("Redis must not be queried before owner verification"))
    monkeypatch.setattr(chatbot, "_require_owned_session", deny_owner)
    monkeypatch.setattr(chatbot, "create_redis_client", redis_factory)
    _set_identity("bob")

    response = TestClient(app).get(f"/api/chat/history/{uuid4()}")

    assert response.status_code == 404
    redis_factory.assert_not_called()
