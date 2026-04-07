from fastapi import APIRouter, HTTPException

from app.schemas.request import ResearchRequest
from app.workflows.rag_workflow import run_chat_workflow

router = APIRouter()


@router.post("/deep-research")
async def deep_research(request: ResearchRequest):
    """
    Run full agentic pipeline without article_id (web search only).
    ReaderAgent automatically skips PGVector when article_id is None.
    """
    try:
        result = run_chat_workflow(
            question=request.query,
            article_id=None,
        )

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def research_status(task_id: str):
    """Placeholder — async task polling via Redis/Celery (to be implemented)."""
    return {
        "task_id": task_id,
        "status":  "pending",
        "message": "Real-time task polling to be implemented via Redis/Celery later.",
    }
