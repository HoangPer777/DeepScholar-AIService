import asyncio
import concurrent.futures
import logging
import uuid
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
        "decision": "accept" if result.get("reviewed_answer") else "rewrite (max iterations reached)",
        "review_feedback":   result.get("review_feedback"),
    }
    if include_timings:
        response["timings"] = result.get("timings", {})
    return response


async def _run_job(task_id: str, question: str, debug: bool = False):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,  # Use dedicated executor, not default (prevents saturation)
            lambda: run_chat_workflow(question=question, article_id=None, session_id=task_id)
        )
        _job_store.update_job(
            task_id,
            {"status": "done", "result": _build_response(result, task_id, include_timings=debug)},
        )
    except Exception as e:
        logger.exception("Deep research job %s failed", task_id)
        _job_store.update_job(task_id, {"status": "error", "error": str(e)})


@router.post("/deep-research")
async def deep_research(request: ResearchRequest):
    """
    Start async deep research job. Returns task_id immediately.
    Client should poll GET /status/{task_id} until status == 'done'.
    """
    task_id = str(uuid.uuid4())
    _job_store.create_job(task_id, {"status": "pending", "debug": request.debug})
    asyncio.create_task(_run_job(task_id, request.query, request.debug))
    return {"task_id": task_id, "status": "pending"}


@router.get("/status/{task_id}")
async def research_status(task_id: str):
    """Poll job status. Returns result when done."""
    job = _job_store.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"])
    if job["status"] == "done":
        # Clean up after delivering result
        result = job["result"]
        _job_store.delete_job(task_id)
        return {"status": "done", **result}
    return {"status": "pending"}
