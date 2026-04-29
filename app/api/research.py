import asyncio
import uuid
from fastapi import APIRouter, HTTPException

from app.schemas.request import ResearchRequest
from app.workflows.rag_workflow import run_chat_workflow

router = APIRouter()

# In-memory job store: task_id -> {"status": "pending"|"done"|"error", "result": ..., "error": ...}
_jobs: dict = {}


def _build_response(result: dict) -> dict:
    raw_sources = [
        s for s in result.get("external_context", [])
        if s.get("title") != "__research_notes__"
    ]
    return {
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


async def _run_job(task_id: str, question: str):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_chat_workflow(question=question, article_id=None)
        )
        _jobs[task_id] = {"status": "done", "result": _build_response(result)}
    except Exception as e:
        _jobs[task_id] = {"status": "error", "error": str(e)}


@router.post("/deep-research")
async def deep_research(request: ResearchRequest):
    """
    Start async deep research job. Returns task_id immediately.
    Client should poll GET /status/{task_id} until status == 'done'.
    """
    task_id = str(uuid.uuid4())
    _jobs[task_id] = {"status": "pending"}
    asyncio.create_task(_run_job(task_id, request.query))
    return {"task_id": task_id, "status": "pending"}


@router.get("/status/{task_id}")
async def research_status(task_id: str):
    """Poll job status. Returns result when done."""
    job = _jobs.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"])
    if job["status"] == "done":
        # Clean up after delivering result
        result = job["result"]
        del _jobs[task_id]
        return {"status": "done", **result}
    return {"status": "pending"}
