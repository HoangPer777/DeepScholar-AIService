import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from fastapi import APIRouter, HTTPException

from app.schemas.request import ResearchRequest
from app.workflows.graph import app as deep_research_graph


router = APIRouter()
_GRAPH_EXECUTOR = ThreadPoolExecutor(max_workers=4)


@router.post("/deep-search")
async def deep_search(request: ResearchRequest):
    started_at = time.time()
    timeout_seconds = int(os.getenv("DEEP_RESEARCH_TIMEOUT_SECONDS", "90"))
    print(
        f"[deep-search] start question='{request.question[:80]}' max_iterations={request.max_iterations} timeout={timeout_seconds}s",
        flush=True,
    )

    initial_state = {
        "question": request.question,
        "max_iterations": request.max_iterations,
    }

    try:
        future = _GRAPH_EXECUTOR.submit(deep_research_graph.invoke, initial_state)
        result = future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        elapsed = round(time.time() - started_at, 2)
        print(f"[deep-search] timeout after {elapsed}s", flush=True)
        raise HTTPException(
            status_code=504,
            detail=(
                f"Deep research timed out after {elapsed}s. "
                "Check logs to see which agent step is slow."
            ),
        )
    except Exception as exc:
        elapsed = round(time.time() - started_at, 2)
        print(f"[deep-search] failed after {elapsed}s error={exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Deep research failed: {exc}")

    answer = result.get("final_answer") or result.get("draft_answer") or ""
    elapsed = round(time.time() - started_at, 2)
    print(
        f"[deep-search] done in {elapsed}s confidence={result.get('confidence_score', 0.0)} iterations={result.get('iteration_count', 0)}",
        flush=True,
    )

    return {
        "question": request.question,
        "elapsed_seconds": elapsed,
        "need_clarification": result.get("need_clarification", False),
        "clarified_question": result.get("clarified_question"),
        "answer": answer,
        "confidence_score": result.get("confidence_score", 0.0),
        "review_feedback": result.get("feedback", ""),
        "iteration_count": result.get("iteration_count", 0),
        "rewrite_required": result.get("rewrite_required", False),
        "research_queries": result.get("research_queries", []),
        "ranked_context": result.get("ranked_context", []),
        "logs": result.get("logs", []),
        "contexts": {
            "vector_context": result.get("vector_context", []),
            "external_context": result.get("external_context", []),
            "memory_context": result.get("memory_context", []),
        },
    }