import time
from typing import List
import os

from langgraph.types import Send

from app.core.llm import get_fast_llm, get_llm
from app.core.memory import memory_recall, memory_save
from app.core.state import AgentState, ClarifyOutput, PlanOutput, ReviewOutput
from app.tools.external_search import external_search
from app.tools.reranker import rerank
from app.tools.vector_search import vector_search


def planner(state: AgentState) -> dict:
    print(f"[agent:planner] start question='{state.question[:80]}'", flush=True)
    try:
        llm = get_llm()
        structured = llm.with_structured_output(PlanOutput)
        result = structured.invoke(
            f"""You are a research planning expert.
Analyze this question and create a research plan:
\"{state.question}\"
- Set need_clarification true only if genuinely ambiguous
- Set need_external_search true if current/recent info is needed
- Generate 3-5 diverse research queries
- Use same language as the question
"""
        )
        return {
            "need_clarification": result.need_clarification,
            "need_external_search": result.need_external_search,
            "research_queries": result.queries,
            "logs": [
                {
                    "agent": "planner",
                    "timestamp": time.time(),
                    "queries": result.queries,
                    "need_clarification": result.need_clarification,
                    "need_external_search": result.need_external_search,
                }
            ],
        }
    except Exception as e:
        return {
            "need_clarification": False,
            "need_external_search": True,
            "research_queries": [state.question],
            "logs": [{"agent": "planner", "error": str(e), "timestamp": time.time()}],
        }
    finally:
        print("[agent:planner] done", flush=True)


def clarifier(state: AgentState) -> dict:
    print("[agent:clarifier] start", flush=True)
    try:
        llm = get_fast_llm()
        structured = llm.with_structured_output(ClarifyOutput)
        result = structured.invoke(
            f"""The question may be unclear:
\"{state.question}\"
Rewrite to be specific, unambiguous, preserve intent, same language.
"""
        )
        return {
            "clarified_question": result.clarified_question,
            "logs": [
                {
                    "agent": "clarifier",
                    "timestamp": time.time(),
                    "clarified": result.clarified_question,
                }
            ],
        }
    except Exception as e:
        return {
            "clarified_question": state.question,
            "logs": [{"agent": "clarifier", "error": str(e), "timestamp": time.time()}],
        }
    finally:
        print("[agent:clarifier] done", flush=True)


def dispatch_node(state: AgentState) -> dict:
    max_queries = int(os.getenv("MAX_RESEARCH_QUERIES", "1"))
    effective_queries = (state.research_queries or [state.clarified_question or state.question])[:max_queries]
    return {
        "logs": [
            {
                "agent": "dispatch",
                "timestamp": time.time(),
                "queries": effective_queries,
                "n_workers": len(effective_queries) * (3 if state.need_external_search else 2),
            }
        ]
    }


def dispatch_router(state: AgentState) -> List[Send]:
    sends: List[Send] = []
    question = state.clarified_question or state.question

    max_queries = int(os.getenv("MAX_RESEARCH_QUERIES", "1"))
    queries = (state.research_queries or [question])[:max_queries]
    print(f"[agent:dispatch] fanout_queries={len(queries)} external={state.need_external_search}", flush=True)
    for q in queries:
        sends.append(Send("reader", {"query": q, "question": question}))
        if state.need_external_search:
            sends.append(Send("researcher", {"query": q, "question": question}))
        sends.append(Send("memory_agent", {"query": q, "question": question}))
    return sends


def reader(state: dict) -> dict:
    query = state["query"]
    print(f"[agent:reader] query='{query[:80]}'", flush=True)

    if os.getenv("ENABLE_VECTOR_SEARCH", "false").lower() != "true":
        return {
            "vector_context": [],
            "logs": [
                {
                    "agent": "reader",
                    "timestamp": time.time(),
                    "query": query,
                    "n_results": 0,
                    "skipped": "ENABLE_VECTOR_SEARCH is false",
                }
            ],
        }

    try:
        results = vector_search(query, top_k=5)
        return {
            "vector_context": results,
            "logs": [{"agent": "reader", "timestamp": time.time(), "query": query, "n_results": len(results)}],
        }
    except Exception as e:
        return {
            "vector_context": [],
            "logs": [{"agent": "reader", "error": str(e), "query": query, "timestamp": time.time()}],
        }


def researcher(state: dict) -> dict:
    query = state["query"]
    print(f"[agent:researcher] query='{query[:80]}'", flush=True)
    try:
        results = external_search(query, top_k=5)
        return {
            "external_context": results,
            "logs": [
                {
                    "agent": "researcher",
                    "timestamp": time.time(),
                    "query": query,
                    "n_results": len(results),
                }
            ],
        }
    except Exception as e:
        return {
            "external_context": [],
            "logs": [{"agent": "researcher", "error": str(e), "query": query, "timestamp": time.time()}],
        }


def memory_agent(state: dict) -> dict:
    query = state["query"]
    print(f"[agent:memory] query='{query[:80]}'", flush=True)
    try:
        results = memory_recall(query=query, top_k=3)
        return {
            "memory_context": results,
            "logs": [
                {
                    "agent": "memory_agent",
                    "timestamp": time.time(),
                    "query": query,
                    "n_results": len(results),
                }
            ],
        }
    except Exception as e:
        return {
            "memory_context": [],
            "logs": [{"agent": "memory_agent", "error": str(e), "query": query, "timestamp": time.time()}],
        }


def ranking(state: AgentState) -> dict:
    print(
        f"[agent:ranking] vector={len(state.vector_context)} external={len(state.external_context)} memory={len(state.memory_context)}",
        flush=True,
    )
    combined = state.vector_context + state.external_context + state.memory_context
    if not combined:
        return {
            "ranked_context": [],
            "logs": [{"agent": "ranking", "timestamp": time.time(), "n_total": 0, "n_ranked": 0}],
        }

    question = state.clarified_question or state.question
    ranked = rerank(question=question, contexts=combined, top_k=7)
    return {
        "ranked_context": ranked,
        "logs": [
            {
                "agent": "ranking",
                "timestamp": time.time(),
                "n_vector": len(state.vector_context),
                "n_external": len(state.external_context),
                "n_memory": len(state.memory_context),
                "n_total": len(combined),
                "n_ranked": len(ranked),
            }
        ],
    }


def writer(state: AgentState) -> dict:
    print(f"[agent:writer] start iteration={state.iteration_count}", flush=True)
    question = state.clarified_question or state.question

    context_parts = []
    for i, ctx in enumerate(state.ranked_context, 1):
        source = ctx.get("source", "unknown")
        title = ctx.get("title", "")
        text = ctx.get("text", "")
        label = f"[{i}] {title} ({source})" if title else f"[{i}] {source}"
        context_parts.append(f"{label}:\n{text}")
    context_str = "\n\n".join(context_parts)

    if not context_str:
        try:
            llm = get_llm()
            fallback_prompt = f"""You are an expert AI assistant.
Question: {question}
No reliable retrieval context is currently available.
- Provide a concise, factual general answer from core knowledge.
- Clearly state that external citations are temporarily unavailable.
- Use same language as the question.
"""
            response = llm.invoke(fallback_prompt)
            draft = response.content if hasattr(response, "content") else str(response)
            return {
                "draft_answer": draft,
                "logs": [{"agent": "writer", "timestamp": time.time(), "ctx_used": 0, "fallback": "llm_no_context"}],
            }
        except Exception:
            return {
                "draft_answer": "Khong tim thay du ngu canh de tra loi. Vui long thu lai voi cau hoi cu the hon.",
                "logs": [{"agent": "writer", "timestamp": time.time(), "ctx_used": 0}],
            }

    feedback_section = ""
    if state.feedback and state.iteration_count > 0:
        feedback_section = f"Previous review feedback:\n{state.feedback}\n"

    prompt = f"""You are an expert research writer.
{feedback_section}
Question: {question}
Research Context:
{context_str}
- Write comprehensive answer
- Cite sources using [1], [2], ...
- Keep factual and avoid unsupported claims
- Use same language as the question
"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        draft = response.content if hasattr(response, "content") else str(response)
        memory_save({"q": question, "a": draft, "timestamp": time.time()})
        return {
            "draft_answer": draft,
            "logs": [
                {
                    "agent": "writer",
                    "timestamp": time.time(),
                    "iteration": state.iteration_count,
                    "is_rewrite": state.iteration_count > 0,
                    "ctx_used": len(state.ranked_context),
                }
            ],
        }
    except Exception as e:
        return {
            "draft_answer": f"Unable to generate answer due to: {str(e)}",
            "logs": [{"agent": "writer", "error": str(e), "timestamp": time.time()}],
        }
    finally:
        print("[agent:writer] done", flush=True)


def reviewer(state: AgentState) -> dict:
    print(f"[agent:reviewer] start iteration={state.iteration_count}", flush=True)
    try:
        llm = get_llm()
        structured = llm.with_structured_output(ReviewOutput)
        result = structured.invoke(
            f"""Evaluate this answer.
QUESTION: {state.question}
ANSWER:
{state.draft_answer}
Score by Accuracy, Completeness, Citations, Clarity.
If score < 0.8 set rewrite_required true and provide actionable feedback.
"""
        )
        return {
            "confidence_score": result.confidence_score,
            "feedback": result.feedback,
            "rewrite_required": result.rewrite_required,
            "iteration_count": state.iteration_count + 1,
            "logs": [
                {
                    "agent": "reviewer",
                    "timestamp": time.time(),
                    "confidence_score": result.confidence_score,
                    "rewrite_required": result.rewrite_required,
                    "iteration": state.iteration_count + 1,
                }
            ],
        }
    except Exception as e:
        return {
            "confidence_score": 0.0,
            "feedback": "Reviewer failed due to LLM/runtime error. Please retry after fixing provider/quota.",
            "rewrite_required": False,
            "iteration_count": state.iteration_count + 1,
            "logs": [{"agent": "reviewer", "error": str(e), "timestamp": time.time(), "llm_failed": True}],
        }
    finally:
        print("[agent:reviewer] done", flush=True)


def planner_router(state: AgentState) -> str:
    if state.need_clarification:
        return "clarifier"
    return "dispatch"


def review_router(state: AgentState) -> str:
    if state.confidence_score > 0.8:
        return "accept"
    if state.iteration_count >= state.max_iterations:
        return "accept"
    if state.rewrite_required:
        return "rewrite"
    return "accept"
