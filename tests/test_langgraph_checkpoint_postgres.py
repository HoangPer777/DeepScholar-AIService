"""Supabase-backed checkpoint smoke test; requires the configured DATABASE_URL."""

from uuid import uuid4

from app.core.graph_checkpoint import close_graph_checkpointer, get_graph_checkpointer
from app.graph import build_graph as graph_module
from app.workflows.states import AgentState

from tests.test_langgraph_parallel_research import _Clarifier, _Planner, _Reader, _Researcher, _Reviewer, _Writer


def test_supabase_checkpoint_persists_graph_state(monkeypatch):
    monkeypatch.setattr(graph_module, "get_safe_llm", lambda _role: None)
    monkeypatch.setattr(graph_module, "PlannerAgent", _Planner)
    monkeypatch.setattr(graph_module, "ClarifierAgent", _Clarifier)
    monkeypatch.setattr(graph_module, "ReaderAgent", _Reader)
    monkeypatch.setattr(graph_module, "ResearcherAgent", _Researcher)
    monkeypatch.setattr(graph_module, "WriterAgent", _Writer)
    monkeypatch.setattr(graph_module, "ReviewerAgent", _Reviewer)

    thread_id = f"test-langgraph-{uuid4()}"
    try:
        graph = graph_module.build_graph(checkpointer=get_graph_checkpointer())
        graph.invoke(AgentState(question="checkpoint smoke test"), config={"configurable": {"thread_id": thread_id}})
        checkpoint = graph.get_state({"configurable": {"thread_id": thread_id}})
        assert checkpoint.values["workflow_status"] == "accepted"
        assert checkpoint.values["reviewed_answer"] == "Grounded draft [1]."
    finally:
        close_graph_checkpointer()


def test_supabase_checkpoint_can_be_inspected_and_resumed(monkeypatch):
    monkeypatch.setattr(graph_module, "get_safe_llm", lambda _role: None)
    monkeypatch.setattr(graph_module, "PlannerAgent", _Planner)
    monkeypatch.setattr(graph_module, "ClarifierAgent", _Clarifier)
    monkeypatch.setattr(graph_module, "ReaderAgent", _Reader)
    monkeypatch.setattr(graph_module, "ResearcherAgent", _Researcher)
    monkeypatch.setattr(graph_module, "WriterAgent", _Writer)
    monkeypatch.setattr(graph_module, "ReviewerAgent", _Reviewer)

    thread_id = f"test-langgraph-resume-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        graph = graph_module.build_graph(checkpointer=get_graph_checkpointer())
        graph.invoke(AgentState(question="resume smoke test"), config=config)
        graph.update_state(config, {"review_feedback": "checkpoint inspected"}, as_node="reviewer")
        resumed = graph.get_state(config)
        assert resumed.values["review_feedback"] == "checkpoint inspected"
        assert resumed.values["reviewed_answer"] == "Grounded draft [1]."
    finally:
        close_graph_checkpointer()
