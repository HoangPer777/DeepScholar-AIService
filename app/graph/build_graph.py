"""Compiled LangGraph orchestration for the grounded Deep Research workflow."""

from typing import Callable, Optional

from langgraph.graph import END, START, StateGraph

from app.agents.clarifier import ClarifierAgent
from app.agents.planner import PlannerAgent
from app.agents.reader import ReaderAgent
from app.agents.researcher import ResearcherAgent
from app.agents.reviewer import ReviewerAgent
from app.agents.writer import WriterAgent
from app.core.config import settings
from app.core.llm import get_safe_llm
from app.workflows.states import AgentState


def _describe_model(llm, agent: str) -> dict:
    """Return serializable model-routing telemetry for graph state."""
    usage_snapshot = getattr(llm, "usage_snapshot", None)
    if callable(usage_snapshot):
        snapshot = dict(usage_snapshot())
        snapshot.setdefault("agent", agent)
        return snapshot

    candidates = getattr(llm, "candidates", None)
    if isinstance(candidates, list) and candidates:
        return {
            "agent": agent,
            "provider": "OpenRouter",
            "model": None,
            "status": "configured",
            "routing": f"OpenRouter candidates · {len(candidates)} available",
            "available_models": [str(candidate) for candidate in candidates],
            "fallback_used": False,
            "invocations": 0,
            "attempts": [],
        }

    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if not isinstance(model, str) or not model.strip():
        model = getattr(settings, "GROQ_LLM_MODEL", "Configured model")
    return {
        "agent": agent,
        "provider": str(getattr(settings, "AGENT_LLM_PROVIDER", "Configured")).title(),
        "model": str(model),
        "selected_provider": None,
        "selected_model": None,
        "routing": "Direct provider",
        "status": "configured",
        "fallback_used": False,
        "invocations": 0,
        "available_models": [],
        "attempts": [],
    }


def _not_called_model(llm, agent: str) -> dict:
    """Describe a configured graph model before its node is invoked."""
    snapshot = _describe_model(llm, agent)
    snapshot.update(
        {
            "status": "not_called",
            "fallback_used": False,
            "invocations": 0,
            "attempts": [],
        }
    )
    return snapshot


def _reader_model_usage() -> dict:
    return {
        "agent": "reader",
        "provider": "Internal retrieval",
        "model": None,
        "selected_provider": None,
        "selected_model": None,
        "status": "not_applicable",
        "routing": "PGVector context",
        "fallback_used": False,
        "invocations": 0,
        "available_models": [],
        "attempts": [],
    }


def _review_router(state: AgentState) -> str:
    if state.reviewed_answer:
        return "accept"
    if state.failure_code:
        return "reject"
    if state.iteration_count < state.max_iterations:
        return "writer"
    return "reject"


def _clarifier_router(state: AgentState) -> str:
    return "clarifier" if state.need_clarification else "prepare_retrieval"


def _evidence_router(state: AgentState) -> str:
    return "writer" if state.evidence_available else "reject"


def build_graph(checkpointer=None, progress_callback: Optional[Callable[[dict], None]] = None):
    """Build the runtime graph; Reader and Researcher execute in one super-step."""
    planner_llm = get_safe_llm("planner")
    clarifier_llm = get_safe_llm("clarifier")
    researcher_llm = get_safe_llm("researcher")
    writer_llm = get_safe_llm("writer")
    reviewer_llm = get_safe_llm("reviewer")
    planner = PlannerAgent(planner_llm)
    clarifier = ClarifierAgent(clarifier_llm)
    reader = ReaderAgent()
    researcher = ResearcherAgent(researcher_llm)
    writer = WriterAgent(writer_llm)
    reviewer = ReviewerAgent(reviewer_llm)

    def emit(**event):
        if progress_callback is None:
            return
        try:
            progress_callback(event)
        except Exception:
            # Progress is observability only and must never fail the graph.
            return

    def planner_node(state: AgentState):
        emit(phase="planning", state="active", agent="planner", title="Planning research", detail="PlannerAgent is analyzing the question and defining the research scope.")
        result = planner.run(state.model_copy(deep=True))
        emit(phase="planning", state="completed", agent="planner", title="Research plan complete", detail="PlannerAgent created web and internal retrieval plans.", metadata={"search_queries": result.search_queries[:5], "focus_sections": result.focus_sections[:10], "need_external_search": True})
        update = result.model_dump()
        update["model_usage"] = {
            "planner": _describe_model(planner_llm, "planner"),
            "clarifier": _not_called_model(clarifier_llm, "clarifier"),
            "reader": _reader_model_usage(),
            "researcher": _not_called_model(researcher_llm, "researcher"),
            "writer": _not_called_model(writer_llm, "writer"),
            "reviewer": _not_called_model(reviewer_llm, "reviewer"),
        }
        return update

    def clarifier_node(state: AgentState):
        emit(phase="clarifying", state="active", agent="clarifier", title="Clarifying the research question", detail="ClarifierAgent is standardizing the interpretation before source discovery.")
        result = clarifier.run(state.model_copy(deep=True))
        emit(phase="clarifying", state="completed", agent="clarifier", title="Research question clarified", detail="The question has been normalized for retrieval.")
        return {
            "clarified_question": result.clarified_question,
            "model_usage": {"clarifier": _describe_model(clarifier_llm, "clarifier")},
        }

    def prepare_retrieval(state: AgentState):
        question = state.clarified_question or state.question
        if not state.need_clarification:
            emit(phase="clarifying", state="skipped", agent="clarifier", title="Question is already clear", detail="No additional clarification was needed.")
        web_queries = state.web_search_queries or state.search_queries or [question]
        db_queries = state.db_search_queries or [question]
        return {
            "need_external_search": True,
            "need_internal_search": True,
            "search_queries": web_queries,
            "web_search_queries": web_queries,
            "db_search_queries": db_queries,
        }

    def reader_node(state: AgentState):
        emit(phase="searching", state="active", agent="reader", title="Searching internal documents", detail="ReaderAgent is searching the selected internal article.")
        result = reader.run(state.model_copy(deep=True))
        emit(phase="searching", state="completed" if result.reader_status == "completed" else result.reader_status, agent="reader", title="Internal document search complete", detail=f"ReaderAgent status: {result.reader_status}.")
        return {
            "vector_context": result.vector_context,
            "reader_status": result.reader_status,
            "retrieval_warnings": result.retrieval_warnings,
            "model_usage": {"reader": _reader_model_usage()},
        }

    def researcher_node(state: AgentState):
        emit(phase="searching", state="active", agent="researcher", title="Searching academic sources", detail=f"ResearcherAgent is processing up to {min(len(state.web_search_queries or state.search_queries), 5)} queries.")
        result = researcher.run(state.model_copy(deep=True), progress_callback=progress_callback)
        emit(phase="searching", state="completed" if result.researcher_status == "completed" else result.researcher_status, agent="researcher", title="Academic source search complete", detail=f"ResearcherAgent status: {result.researcher_status}.")
        return {
            "external_context": result.external_context,
            "researcher_status": result.researcher_status,
            "retrieval_warnings": result.retrieval_warnings,
            "model_usage": {"researcher": _describe_model(researcher_llm, "researcher")},
        }

    def merge_evidence(state: AgentState):
        sources = [source for source in state.external_context if source.get("title") != "__research_notes__"]
        manifest = [
            {"kind": "external", "index": index + 1, "title": item.get("title"), "url": item.get("url")}
            for index, item in enumerate(sources)
        ] + [
            {"kind": "pdf", "index": index + 1, "chunk_id": item.get("chunk_id")}
            for index, item in enumerate(state.vector_context)
        ]
        available = bool(manifest)
        return {
            "evidence_manifest": manifest,
            "evidence_available": available,
            "workflow_status": "evidence_ready" if available else "failed",
            "failure_code": None if available else "insufficient_grounded_evidence",
            "failure_message": None if available else "No internal or external evidence was found.",
        }

    def writer_node(state: AgentState):
        draft_iteration = state.iteration_count + 1
        emit(phase="drafting", state="active", agent="writer", title=f"Writing draft {draft_iteration}", detail="WriterAgent is synthesizing the collected evidence.", iteration=draft_iteration, max_iterations=state.max_iterations)
        result = writer.run(state.model_copy(deep=True))
        draft_history = [*state.draft_history, {
            "iteration": draft_iteration,
            "content": result.draft_answer or "",
        }]
        emit(phase="drafting", state="completed", agent="writer", title=f"Draft {draft_iteration} complete", detail="WriterAgent produced a grounded draft.", iteration=draft_iteration, max_iterations=state.max_iterations)
        return {
            "draft_answer": result.draft_answer,
            "draft_history": draft_history,
            "failure_code": result.failure_code,
            "failure_message": result.failure_message,
            "model_usage": {"writer": _describe_model(writer_llm, "writer")},
            "writer_model": _describe_model(writer_llm, "writer"),
        }

    def reviewer_node(state: AgentState):
        emit(phase="reviewing", state="active", agent="reviewer", title=f"Reviewing draft {state.iteration_count + 1}", detail="ReviewerAgent is checking grounding and citation quality.", iteration=state.iteration_count + 1, max_iterations=state.max_iterations)
        result = reviewer.run(state.model_copy(deep=True))
        decision = "accept" if result.reviewed_answer else ("rejected" if result.iteration_count >= result.max_iterations else "rewrite")
        review_history = [*state.review_history, {
            "iteration": result.iteration_count,
            "score": result.confidence_score,
            "decision": decision,
            "feedback": result.review_feedback or "",
        }]
        emit(phase="reviewing", state="completed", agent="reviewer", title=f"Draft {result.iteration_count} review complete", detail=result.review_feedback or f"Reviewer decision: {decision}.", iteration=result.iteration_count, max_iterations=result.max_iterations, metadata={"decision": decision, "score": result.confidence_score})
        if decision == "rewrite":
            emit(phase="rewriting", state="active", agent="writer", title=f"Rewriting draft {result.iteration_count + 1}", detail="WriterAgent will address reviewer feedback.", iteration=result.iteration_count, max_iterations=result.max_iterations)
        return {
            "reviewed_answer": result.reviewed_answer,
            "review_feedback": result.review_feedback,
            "confidence_score": result.confidence_score,
            "iteration_count": result.iteration_count,
            "review_decision": result.review_decision,
            "review_history": review_history,
            "failure_code": result.failure_code,
            "model_usage": {"reviewer": _describe_model(reviewer_llm, "reviewer")},
        }

    def accept_node(state: AgentState):
        emit(phase="finalizing", state="active", agent="system", title="Finalizing the report", detail="DeepScholar is packaging the accepted report and sources.")
        emit(phase="completed", state="completed", agent="system", title="Research report complete", detail="The reviewed research report is ready.", iteration=state.iteration_count, max_iterations=state.max_iterations)
        return {"workflow_status": "accepted", "review_decision": "accept"}

    def reject_node(state: AgentState):
        emit(phase="failed", state="failed", agent="system", title="Research report rejected", detail=state.failure_message or "The report could not be verified.")
        return {
            "workflow_status": "rejected",
            "review_decision": "rejected",
            "failure_code": state.failure_code or "research_review_rejected",
        }

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("clarifier", clarifier_node)
    graph.add_node("prepare_retrieval", prepare_retrieval)
    graph.add_node("reader", reader_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("merge_evidence", merge_evidence)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("accept", accept_node)
    graph.add_node("reject", reject_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", _clarifier_router, {"clarifier": "clarifier", "prepare_retrieval": "prepare_retrieval"})
    graph.add_edge("clarifier", "prepare_retrieval")
    graph.add_edge("prepare_retrieval", "reader")
    graph.add_edge("prepare_retrieval", "researcher")
    graph.add_edge(["reader", "researcher"], "merge_evidence")
    graph.add_conditional_edges("merge_evidence", _evidence_router, {"writer": "writer", "reject": "reject"})
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", _review_router, {"writer": "writer", "accept": "accept", "reject": "reject"})
    graph.add_edge("accept", END)
    graph.add_edge("reject", END)
    return graph.compile(checkpointer=checkpointer)
