import fakeredis
import pytest

from app.core.job_store import RedisJobStore


def _store():
    store = RedisJobStore.__new__(RedisJobStore)
    store._fallback = {}
    store._fallback_lock = __import__("threading").Lock()
    store._use_redis = True
    store._redis = fakeredis.FakeRedis(decode_responses=True)
    return store


def test_pdf_job_updates_are_atomic_merges_and_preserve_identity():
    store = _store()
    store.create_job_if_absent("pdf-1", {
        "kind": "pdf",
        "task_id": "pdf-1",
        "article_id": 17,
        "generation": 3,
        "content_revision": 2,
        "idempotency_key": "same-key",
        "slug": "paper",
        "status": "queued",
    })

    updated = store.update_pdf_job("pdf-1", {"status": "running", "progress": {"phase": "embedding"}})

    assert updated["status"] == "running"
    assert updated["progress"] == {"phase": "embedding"}
    assert updated["article_id"] == 17
    assert updated["generation"] == 3
    assert store.get_pdf_job("pdf-1") == updated


@pytest.mark.parametrize("terminal", ["completed", "failed"])
def test_pdf_terminal_status_cannot_be_overwritten_by_late_progress(terminal):
    store = _store()
    store.create_job_if_absent("pdf-terminal", {
        "kind": "pdf",
        "task_id": "pdf-terminal",
        "status": terminal,
        "error_code": "preserve-me",
    })

    result = store.update_pdf_job("pdf-terminal", {"status": "running", "progress": {"phase": "late"}})

    assert result["status"] == terminal
    assert result["error_code"] == "preserve-me"
    assert "progress" not in result


def test_pdf_status_does_not_read_non_pdf_jobs():
    store = _store()
    store.create_job("chat-1", {"kind": "chat", "status": "done"})

    assert store.get_pdf_job("chat-1") is None
    assert store.update_pdf_job("chat-1", {"status": "failed"}) is None
    assert store.get_job("chat-1")["status"] == "done"


def test_pdf_status_fails_closed_when_redis_is_not_persistent():
    store = _store()
    store._use_redis = False

    with pytest.raises(RuntimeError, match="Persistent PDF job storage"):
        store.get_pdf_job("pdf-unavailable")
