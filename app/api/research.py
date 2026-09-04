import asyncio
import copy
import concurrent.futures
import logging
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException

from app.core.job_store import get_job_store
from app.schemas.request import ResearchRequest
from app.workflows.rag_workflow import run_chat_workflow

router = APIRouter()
logger = logging.getLogger(__name__)

# Dedicated thread pool for long-running research jobs
# Prevents default ThreadPoolExecutor saturation from blocking the event loop
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Persistent job store backed by Redis (falls back to in-memory when Redis is unavailable)
# Requirements: 2.1, 2.5, 7.1
_job_store = get_job_store()

_PROGRESS_PHASES = {
    "queued",
    "planning",
    "clarifying",
    "searching",
    "synthesizing",
    "drafting",
    "reviewing",
    "rewriting",
    "finalizing",
    "completed",
    "failed",
}
_ACTIVITY_STATES = {"active", "completed", "skipped", "failed"}
_RESEARCH_AGENTS = {
    "system",
    "planner",
    "clarifier",
    "researcher",
    "reader",
    "writer",
    "reviewer",
    "fast_chat",
}
_MAX_ACTIVITIES = 50
_MAX_SOURCE_PREVIEWS = 20
_MAX_DETAIL_LENGTH = 500


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _truncate(value, limit: int = _MAX_DETAIL_LENGTH) -> str:
    return str(value or "").strip()[:limit]


def _sanitize_metadata(value, depth: int = 0):
    """Keep progress metadata small, JSON-safe, and free of runtime objects."""
    if depth > 2 or value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, (list, tuple)):
        return [
            cleaned
            for item in list(value)[:20]
            if (cleaned := _sanitize_metadata(item, depth + 1)) is not None
        ]
    if isinstance(value, dict):
        cleaned = {}
        for key, item in list(value.items())[:20]:
            sanitized = _sanitize_metadata(item, depth + 1)
            if sanitized is not None:
                cleaned[_truncate(key, 80)] = sanitized
        return cleaned
    return None


def _sanitize_source_preview(source: dict) -> dict | None:
    url = _truncate(source.get("url"), 2000)
    if not url.startswith(("http://", "https://")):
        return None
    parsed = urlparse(url)
    domain = _truncate(source.get("domain") or parsed.netloc, 160)
    return {
        "title": _truncate(source.get("title") or domain or "Untitled source", 240),
        "url": url,
        "domain": domain,
        "source_type": _truncate(source.get("source_type") or "web", 80),
        "year": _truncate(source.get("year"), 20) or None,
    }


def _sanitize_model_info(value: dict | None) -> dict | None:
    """Keep model telemetry explicit while excluding provider internals."""
    if not isinstance(value, dict):
        return None

    cleaned = {
        "agent": _truncate(value.get("agent"), 40) or None,
        "provider": _truncate(value.get("provider"), 80) or "Unknown",
        "model": _truncate(value.get("model"), 180) or None,
        "selected_provider": _truncate(value.get("selected_provider"), 80) or None,
        "selected_model": _truncate(value.get("selected_model"), 180) or None,
        "status": _truncate(value.get("status"), 40) or "unknown",
        "routing": _truncate(value.get("routing"), 120) or None,
        "fallback_used": bool(value.get("fallback_used", False)),
        "invocations": max(0, int(value.get("invocations", 0) or 0)),
        "available_models": [
            _truncate(model, 180)
            for model in list(value.get("available_models") or [])[:8]
            if _truncate(model, 180)
        ],
        "attempts": [],
    }
    for attempt in list(value.get("attempts") or [])[:8]:
        if not isinstance(attempt, dict):
            continue
        cleaned["attempts"].append(
            {
                "model": _truncate(attempt.get("model"), 180),
                "provider": _truncate(attempt.get("provider"), 80),
                "status": _truncate(attempt.get("status"), 40),
                "error_type": _truncate(attempt.get("error_type"), 60) or None,
                "duration_ms": max(0, int(attempt.get("duration_ms", 0) or 0)),
            }
        )
    return cleaned


def _initial_job_snapshot(debug: bool = False) -> dict:
    now = _utc_now()
    return {
        "status": "pending",
        "debug": debug,
        "progress": {
            "phase": "queued",
            "message": "Research request accepted",
            "iteration": 0,
            "max_iterations": 2,
            "started_at": now,
            "updated_at": now,
        },
        "activities": [
            {
                "sequence": 1,
                "phase": "queued",
                "state": "completed",
                "agent": "system",
                "title": "Research request accepted",
                "detail": "DeepScholar is preparing the research pipeline.",
                "timestamp": now,
                "metadata": {},
            }
        ],
        "source_previews": [],
    }


class _ProgressEmitter:
    """Maintain one bounded job snapshot and persist it without extra Redis reads."""

    def __init__(self, task_id: str, initial_snapshot: dict):
        self.task_id = task_id
        self._snapshot = copy.deepcopy(initial_snapshot)
        self._sequence = max(
            (item.get("sequence", 0) for item in self._snapshot.get("activities", [])),
            default=0,
        )

    @property
    def snapshot(self) -> dict:
        return copy.deepcopy(self._snapshot)

    def emit(self, event: dict) -> None:
        try:
            phase = event.get("phase")
            if phase not in _PROGRESS_PHASES:
                logger.warning("Ignoring unknown research progress phase: %s", phase)
                return

            activity_state = event.get("state", "active")
            if activity_state not in _ACTIVITY_STATES:
                activity_state = "active"

            agent = event.get("agent", "system")
            if agent not in _RESEARCH_AGENTS:
                agent = "system"

            now = _utc_now()
            current_progress = self._snapshot.get("progress", {})
            iteration = event.get("iteration", current_progress.get("iteration", 0))
            max_iterations = event.get(
                "max_iterations", current_progress.get("max_iterations", 2)
            )
            message = _truncate(event.get("message") or event.get("title"))

            self._snapshot["progress"] = {
                "phase": phase,
                "message": message,
                "iteration": max(0, int(iteration or 0)),
                "max_iterations": max(1, int(max_iterations or 1)),
                "started_at": current_progress.get("started_at", now),
                "updated_at": now,
            }

            self._sequence += 1
            activity = {
                "sequence": self._sequence,
                "phase": phase,
                "state": activity_state,
                "agent": agent,
                "title": _truncate(event.get("title") or message, 200),
                "detail": _truncate(event.get("detail")),
                "timestamp": now,
                "metadata": _sanitize_metadata(event.get("metadata") or {}) or {},
            }
            activities = [*self._snapshot.get("activities", []), activity]
            self._snapshot["activities"] = activities[-_MAX_ACTIVITIES:]

            previews_by_url = {
                item.get("url"): item
                for item in self._snapshot.get("source_previews", [])
                if item.get("url")
            }
            for source in event.get("source_previews") or []:
                if not isinstance(source, dict):
                    continue
                preview = _sanitize_source_preview(source)
                if preview and preview["url"] not in previews_by_url:
                    previews_by_url[preview["url"]] = preview
            self._snapshot["source_previews"] = list(previews_by_url.values())[
                :_MAX_SOURCE_PREVIEWS
            ]

            _job_store.update_job(self.task_id, copy.deepcopy(self._snapshot))
        except Exception:
            # Progress is observability data and must never fail the research job.
            logger.warning(
                "Unable to persist progress for deep research job %s",
                self.task_id,
                exc_info=True,
            )


class _JobStoreMappingAdapter:
    """Backward-compatible mapping facade for legacy tests/imports.

    Runtime code uses RedisJobStore directly. This adapter keeps old code that
    injects ``app.api.research._jobs[task_id] = data`` working without
    reintroducing a separate in-memory source of truth.
    """

    def __setitem__(self, task_id: str, data: dict) -> None:
        _job_store.create_job(task_id, data)

    def __getitem__(self, task_id: str) -> dict:
        job = _job_store.get_job(task_id)
        if job is None:
            raise KeyError(task_id)
        return job

    def __contains__(self, task_id: object) -> bool:
        return isinstance(task_id, str) and _job_store.get_job(task_id) is not None

    def get(self, task_id: str, default=None):
        job = _job_store.get_job(task_id)
        return default if job is None else job

    def pop(self, task_id: str, default=None):
        job = _job_store.get_job(task_id)
        if job is None:
            return default
        _job_store.delete_job(task_id)
        return job


_jobs = _JobStoreMappingAdapter()


def _build_response(result: dict, task_id: str, include_timings: bool = False) -> dict:
    raw_sources = [
        s for s in result.get("external_context", [])
        if s.get("title") != "__research_notes__"
    ]
    response = {
        "session_id": task_id,
        "answer": result.get("reviewed_answer") or result.get("draft_answer") or "",
        "sources": [
            {
                "index":       i + 1,
                "title":       s.get("title", ""),
                "url":         s.get("url", ""),
                "score":       s.get("score", 0.0),
                "source_type": s.get("source_type", "web"),
                "apa_year":    s.get("apa_year", "n.d."),
                "apa_authors": s.get("apa_authors"),
                "apa_venue":   s.get("apa_venue"),
            }
            for i, s in enumerate(raw_sources)
        ],
        "planner_decision": {
            "need_clarification":   result.get("need_clarification"),
            "need_external_search": result.get("need_external_search"),
            "focus_sections":       result.get("focus_sections"),
            "search_queries":       result.get("search_queries"),
            "clarified_question":   result.get("clarified_question"),
        },
        "confidence_score":  result.get("confidence_score", 0.0),
        "iterations_used":   result.get("iteration_count", 0),
        "writer_model":       _sanitize_model_info(result.get("writer_model")),
        "model_usage": {
            str(agent): _sanitize_model_info(info)
            for agent, info in list((result.get("model_usage") or {}).items())[:8]
            if _sanitize_model_info(info) is not None
        },
        "decision": "accept" if result.get("reviewed_answer") else "rejected",
        "review_feedback":   result.get("review_feedback"),
    }
    if include_timings:
        response["timings"] = result.get("timings", {})
    return response


async def _run_job(
    task_id: str,
    question: str,
    debug: bool = False,
    initial_snapshot: dict | None = None,
):
    emitter = _ProgressEmitter(
        task_id,
        initial_snapshot or _initial_job_snapshot(debug),
    )
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,  # Use dedicated executor, not default (prevents saturation)
            lambda: run_chat_workflow(
                question=question,
                article_id=None,
                session_id=task_id,
                progress_callback=emitter.emit,
            ),
        )
        response = _build_response(result, task_id, include_timings=debug)
        snapshot = emitter.snapshot
        response.update(
            {
                "progress": snapshot.get("progress"),
                "activities": snapshot.get("activities", []),
                "source_previews": snapshot.get("source_previews", []),
            }
        )
        snapshot.update({"status": "done", "result": response})
        _job_store.update_job(task_id, snapshot)
    except Exception as e:
        logger.exception("Deep research job %s failed", task_id)
        error_text = str(e).lower()
        retryable = "429" in error_text or "rate" in error_text or "timeout" in error_text
        current_phase = emitter.snapshot.get("progress", {}).get("phase", "failed")
        if "429" in error_text or "rate" in error_text:
            public_error = "rate_limit_exceeded: AI providers are busy. Please try again later."
        elif "timeout" in error_text:
            public_error = "Deep research timed out. Please try again."
        else:
            public_error = f"Deep research failed during {current_phase}."

        emitter.emit(
            {
                "phase": "failed",
                "state": "failed",
                "title": "Unable to complete research",
                "detail": f"Pipeline stopped during {current_phase}.",
                "message": "Research stopped due to an error",
                "metadata": {"failed_phase": current_phase},
            }
        )
        snapshot = emitter.snapshot
        snapshot.update(
            {
                "status": "error",
                "error_code": "deep_research_failed",
                "error": public_error,
                "retryable": retryable,
            }
        )
        _job_store.update_job(task_id, snapshot)


@router.post("/deep-research")
async def deep_research(request: ResearchRequest):
    """
    Start async deep research job. Returns task_id immediately.
    Client should poll GET /status/{task_id} until status == 'done'.
    """
    task_id = str(uuid.uuid4())
    snapshot = _initial_job_snapshot(request.debug)
    _job_store.create_job(task_id, snapshot)
    asyncio.create_task(_run_job(task_id, request.query, request.debug, snapshot))
    return {"task_id": task_id, "status": "pending"}


@router.get("/status/{task_id}")
async def research_status(task_id: str):
    """Poll job status. Returns result when done."""
    job = _job_store.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    if job["status"] == "error":
        return {
            "status": "error",
            "error_code": job.get("error_code", "deep_research_failed"),
            "message": job.get("error", "Deep research failed."),
            "retryable": job.get("retryable", True),
            "progress": job.get("progress"),
            "activities": job.get("activities", []),
            "source_previews": job.get("source_previews", []),
        }
    if job["status"] == "done":
        # Clean up after delivering result
        result = job["result"]
        _job_store.delete_job(task_id)
        return {"status": "done", **result}
    pending = {"status": "pending"}
    if job.get("progress"):
        pending.update(
            {
                "progress": job["progress"],
                "activities": job.get("activities", []),
                "source_previews": job.get("source_previews", []),
            }
        )
    return pending
