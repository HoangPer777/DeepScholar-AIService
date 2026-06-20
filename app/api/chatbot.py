"""
Chatbot API endpoints.

Implements async job pattern for follow-up chat (same as research.py):
  POST /api/chat/start → returns task_id immediately
  GET  /api/chat/status/{task_id} → poll until done

Also implements dual-write pattern for chat history persistence:
  POST /api/chat/ → (legacy sync endpoint, kept for backward compat)
  GET  /api/chat/history/{session_id} → try Redis first → fallback to PostgreSQL
"""

import asyncio
import concurrent.futures
import logging
import uuid

import redis as redis_lib
from fastapi import APIRouter, HTTPException

from app.core.database import SessionLocal
from app.core.job_store import get_job_store
from app.core.memory_store import MemoryStore, SessionContextNotFoundError
from app.core.message_store import MessageStore
from app.core.redis_client import create_redis_client
from app.schemas.chat_models import Message, MessageRole
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.workflows.rag_workflow import run_chat_workflow

logger = logging.getLogger(__name__)
router = APIRouter()

_message_store = MessageStore()

# Dedicated thread pool — prevents event loop blocking during long LLM calls
# Shared with research.py pattern for consistency
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Persistent job store backed by Redis (falls back to in-memory when Redis is unavailable)
# Requirements: 2.1, 2.5, 3.5, 7.2
_job_store = get_job_store()


# ---------------------------------------------------------------------------
# Async job pattern (used by follow-up chat to avoid proxy timeout)
# ---------------------------------------------------------------------------

def _answer_from_result(result: dict) -> str:
    return result.get("answer") or result.get("reviewed_answer") or result.get("draft_answer") or ""


def _citations_from_result(result: dict) -> list:
    if result.get("is_fast_chat") and isinstance(result.get("citations"), list):
        return result["citations"]
    return _extract_citations(result)


def _build_chat_response(result: dict, session_id: str, article_id, include_timings: bool = False) -> dict:
    """Build serializable chat response dict from workflow result."""
    answer = _answer_from_result(result)
    citations = _citations_from_result(result)
    response = {
        "session_id": result.get("session_id") or session_id,
        "article_id": article_id,
        "answer": answer,
        "citations": citations,
        "confidence_score": result.get("confidence_score", 0.0),
        "review_feedback": result.get("review_feedback"),
        "need_clarification": result.get("need_clarification", False),
        "clarification_question": result.get("clarified_question"),
        "is_fast_chat": result.get("is_fast_chat", False),
    }
    if include_timings:
        response["timings"] = result.get("timings", {})
    return response


def _has_research_context(session_id: str) -> bool:
    redis_client = create_redis_client(from_url=redis_lib.from_url)
    try:
        store = MemoryStore(redis_client)
        context = store.get_context_window(session_id)
        return context.research_report is not None
    finally:
        redis_client.close()


async def _run_chat_job(
    task_id: str,
    question: str,
    article_id,
    session_id: str,
    require_context: bool = False,
    debug: bool = False,
):
    """Run chat workflow in thread pool and store result in _chat_jobs."""
    try:
        if require_context and not _has_research_context(session_id):
            raise SessionContextNotFoundError(
                f"Session '{session_id}' not found or has no research context."
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: run_chat_workflow(
                question=question,
                article_id=article_id,
                session_id=session_id,
            ),
        )

        answer = _answer_from_result(result)
        citations = _citations_from_result(result)

        # Dual-write after workflow completes
        _persist_to_postgres(
            session_id=session_id,
            question=question,
            answer=answer,
            confidence_score=result.get("confidence_score", 0.0),
            citations=citations,
        )
        if not result.get("is_fast_chat"):
            _persist_to_redis(session_id=session_id, question=question, answer=answer)

        _job_store.update_job(task_id, {
            "status": "done",
            "result": _build_chat_response(result, session_id, article_id, include_timings=debug),
        })
    except SessionContextNotFoundError:
        # Requirement 3.5: return 404 when session has no research context
        logger.warning("Chat job %s: session '%s' not found or has no research context", task_id, session_id)
        _job_store.update_job(task_id, {
            "status": "error",
            "error": f"Session '{session_id}' not found or has no research context. Please start a deep research session first.",
            "error_code": 404,
        })
    except Exception as e:
        logger.error("Chat job %s failed: %s", task_id, e)
        _job_store.update_job(task_id, {"status": "error", "error": str(e)})


@router.post("/start")
async def chat_start(request: ChatRequest):
    """
    Start async chat job. Returns task_id immediately.
    Client should poll GET /status/{task_id} until status == 'done'.

    This avoids Vercel serverless proxy timeout (60s) for long LLM workflows.
    """
    task_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    require_context = request.session_id is not None

    _job_store.create_job(task_id, {"status": "pending", "session_id": session_id, "debug": request.debug})
    asyncio.create_task(
        _run_chat_job(
            task_id,
            request.question,
            request.article_id,
            session_id,
            require_context=require_context,
            debug=request.debug,
        )
    )
    return {"task_id": task_id, "session_id": session_id, "status": "pending"}


@router.get("/status/{task_id}")
async def chat_status(task_id: str):
    """Poll chat job status. Returns result when done."""
    job = _job_store.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Chat task not found")
    if job["status"] == "error":
        status_code = job.get("error_code", 500)
        raise HTTPException(status_code=status_code, detail=job["error"])
    if job["status"] == "done":
        result = job["result"]
        _job_store.delete_job(task_id)
        return {"status": "done", **result}
    # Return session_id in pending response so frontend can track it
    return {"status": "pending", "session_id": job.get("session_id")}


@router.post("/")
async def chat(request: ChatRequest):
    """
    Run full agentic pipeline: Planner → Clarifier → Researcher → Reader → Writer → Reviewer.

    Dual-write: saves messages to Redis (short-term, TTL 24h) AND PostgreSQL (permanent).
    Runs workflow in a thread executor to avoid blocking the FastAPI event loop.
    """
    try:
        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Run blocking LLM workflow in thread pool — keeps event loop free for
        # concurrent polling requests and other follow-up calls
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: run_chat_workflow(
                question=request.question,
                article_id=request.article_id,
                session_id=session_id,
            ),
        )

        answer = _answer_from_result(result)
        citations = _citations_from_result(result)

        # --- Dual-write: persist to PostgreSQL ---
        _persist_to_postgres(
            session_id=session_id,
            question=request.question,
            answer=answer,
            confidence_score=result.get("confidence_score", 0.0),
            citations=citations,
        )

        # --- Dual-write: persist to Redis (short-term cache) ---
        if not result.get("is_fast_chat"):
            _persist_to_redis(
                session_id=session_id,
                question=request.question,
                answer=answer,
            )

        response_payload = ChatResponse(
            session_id=result.get("session_id") or session_id,
            article_id=request.article_id,
            answer=answer,
            citations=citations,
            confidence_score=result.get("confidence_score", 0.0),
            review_feedback=result.get("review_feedback"),
            need_clarification=result.get("need_clarification", False),
            clarification_question=result.get("clarified_question"),
            timings=result.get("timings") if request.debug else None,
        ).model_dump(exclude_none=True)
        response_payload["article_id"] = request.article_id
        return response_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def chat_history(session_id: str):
    """
    Return chat history for a session.

    Strategy: try Redis first (fast, TTL 24h) → fallback to PostgreSQL (permanent).
    """
    # --- Try Redis first ---
    redis_client = create_redis_client(from_url=redis_lib.from_url)
    try:
        store = MemoryStore(redis_client)
        context = store.get_context_window(session_id)
        history = [msg.model_dump(mode="json") for msg in context.messages]
        if history:
            return {"session_id": session_id, "history": history, "source": "redis"}
    except SessionContextNotFoundError:
        pass  # Fall through to PostgreSQL
    except Exception as exc:
        logger.warning("Redis history lookup failed for session %s: %s", session_id, exc)
    finally:
        redis_client.close()

    try:
        uuid.UUID(session_id)
    except ValueError:
        return {"session_id": session_id, "history": [], "source": "invalid_session_id"}

    # --- Fallback: PostgreSQL (survives Docker restarts) ---
    db = SessionLocal()
    try:
        db_messages = _message_store.get_messages(db, session_id)
        if not db_messages:
            return {"session_id": session_id, "history": [], "source": "postgres"}

        history = [
            {
                "message_id": str(m.message_id),
                "session_id": str(m.session_id),
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "token_count": m.token_count or 0,
                "metadata": m.msg_metadata or {},
            }
            for m in db_messages
        ]
        return {"session_id": session_id, "history": history, "source": "postgres"}
    except Exception as exc:
        logger.error("PostgreSQL history lookup failed for session %s: %s", session_id, exc)
        return {"session_id": session_id, "history": []}
    finally:
        db.close()


@router.get("/sessions")
async def list_sessions(user_id: str = "anonymous"):
    """
    List all chat sessions for a user from PostgreSQL.

    Query param: user_id (default: "anonymous")
    """
    from app.core.session_models import Session as SessionORM
    from sqlalchemy import desc

    db = SessionLocal()
    try:
        sessions = (
            db.query(SessionORM)
            .filter(SessionORM.user_id == user_id)
            .order_by(desc(SessionORM.created_at))
            .limit(50)
            .all()
        )
        return {
            "user_id": user_id,
            "sessions": [
                {
                    "session_id": str(s.session_id),
                    "initial_query": s.initial_query,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    "status": s.status,
                    "message_count": s.message_count,
                }
                for s in sessions
            ],
        }
    except Exception as exc:
        logger.error("Failed to list sessions for user %s: %s", user_id, exc)
        return {"user_id": user_id, "sessions": []}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _persist_to_postgres(
    session_id: str,
    question: str,
    answer: str,
    confidence_score: float,
    citations: list,
) -> None:
    """Save user message + assistant response to PostgreSQL. Fire-and-forget."""
    db = SessionLocal()
    try:
        # Ensure session row exists (creates if missing)
        _message_store.ensure_session_exists(
            db=db,
            session_id=session_id,
            user_id="anonymous",  # Replace with real user_id when auth is added
            initial_query=question,
        )

        # Save user message
        _message_store.save_message(
            db=db,
            session_id=session_id,
            role="user",
            content=question,
        )

        # Save assistant response with metadata
        _message_store.save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            content=answer,
            metadata={
                "confidence_score": confidence_score,
                "citations": citations,
            },
        )
    except Exception as exc:
        # Non-fatal: log and continue — don't fail the chat response
        logger.error(
            "Failed to persist chat to PostgreSQL for session %s: %s",
            session_id,
            exc,
        )
    finally:
        db.close()


def _persist_to_redis(session_id: str, question: str, answer: str) -> None:
    """Save user message + assistant response to Redis MemoryStore. Fire-and-forget."""
    redis_client = create_redis_client(from_url=redis_lib.from_url)
    try:
        store = MemoryStore(redis_client)

        # Init context if this is a new session
        if not store.session_exists(session_id):
            store.init_session_context(session_id)

        # Save user message
        store.save_message(
            session_id,
            Message(
                session_id=session_id,
                role=MessageRole.USER,
                content=question,
            ),
        )

        # Save assistant response
        store.save_message(
            session_id,
            Message(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=answer,
            ),
        )
    except Exception as exc:
        # Non-fatal: log and continue
        logger.warning(
            "Failed to persist chat to Redis for session %s: %s",
            session_id,
            exc,
        )
    finally:
        redis_client.close()


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
