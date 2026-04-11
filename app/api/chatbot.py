from fastapi import APIRouter, HTTPException

from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.workflows.rag_workflow import run_chat_workflow

router = APIRouter()


@router.post("/")
async def chat(request: ChatRequest):
    """
    Run full agentic pipeline: Planner → Clarifier → Researcher → Reader → Writer → Reviewer.
    """
    try:
        result = run_chat_workflow(
            question=request.question,
            article_id=request.article_id,
            session_id=request.session_id,
        )
        return ChatResponse(
            session_id=result.get("session_id") or request.session_id,
            article_id=request.article_id,
            answer=result.get("reviewed_answer") or result.get("draft_answer") or "",
            citations=_extract_citations(result),
            confidence_score=result.get("confidence_score", 0.0),
            review_feedback=result.get("review_feedback"),
            need_clarification=result.get("need_clarification", False),
            clarification_question=result.get("clarified_question"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def chat_history(session_id: str):
    """Placeholder — chat history via session."""
    return {"session_id": session_id, "history": []}


def _extract_citations(result: dict) -> list:
    """Extract raw sources from external_context as citation list."""
    external = result.get("external_context", [])
    return [
        {
            "index": i + 1,
            "title": s.get("title", ""),
            "url":   s.get("url", ""),
            "score": s.get("score", 0.0),
            "source_type": s.get("source_type", "web"),
            "apa_year": s.get("apa_year", "n.d."),
        }
        for i, s in enumerate(external)
        if s.get("title") != "__research_notes__"
    ]
