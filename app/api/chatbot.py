from fastapi import APIRouter

from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse


router = APIRouter()


@router.post("/ask")
async def ask_question(request: ChatRequest):
    """
    TODO: Process user question through multi-agent workflow
    1. Execute LangGraph workflow
    2. Return streaming response with answer and citations
    """
    # TODO: Implementation
    return ChatResponse(
        session_id=request.session_id,
        article_id=request.article_id,
        answer="",
        citations=[],
        confidence_score=0.0,
    )


@router.get("/history/{session_id}")
async def chat_history(session_id: str):
    """
    TODO: Retrieve chat history for session
    """
    # TODO: Implementation
    return {"session_id": session_id, "history": []}
