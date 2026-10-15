import json

from app.graph import build_graph as graph_module
from app.workflows import rag_workflow
from app.workflows.states import AgentState


class _TelemetryLLM:
    def __init__(self, agent):
        self.agent = agent

    def usage_snapshot(self):
        return {
            "agent": self.agent,
            "provider": "OpenRouter",
            "model": f"test/{self.agent}",
            "selected_provider": "OpenRouter",
            "selected_model": f"test/{self.agent}",
            "status": "selected",
            "routing": "OpenRouter candidates",
            "fallback_used": False,
            "invocations": 1,
            "available_models": [f"test/{self.agent}"],
            "attempts": [{
                "model": f"test/{self.agent}",
                "provider": "OpenRouter",
                "status": "success",
                "duration_ms": 1,
            }],
        }


class _Planner:
    def __init__(self, _llm): pass
    def run(self, state):
        state.need_external_search = True
        state.search_queries = [state.question]
        state.web_search_queries = [state.question]
        state.db_search_queries = [state.question]
        return state


class _Clarifier:
    def __init__(self, _llm): pass
    def run(self, state): return state


class _Reader:
    def run(self, state):
        state.reader_status = "skipped" if not state.article_id else "completed"
        return state


class _Researcher:
    def __init__(self, _llm): pass
    def run(self, state, progress_callback=None):
        state.researcher_status = "completed"
        state.external_context = [{"title": "Source", "url": "https://example.com", "source_type": "arxiv"}]
        return state


class _Writer:
    def __init__(self, _llm): pass
    def run(self, state):
        state.draft_answer = "Grounded draft [1]."
        return state


class _Reviewer:
    def __init__(self, _llm): pass
    def run(self, state):
        state.reviewed_answer = state.draft_answer
        state.review_decision = "accept"
        state.confidence_score = 0.9
        return state


def test_graph_runs_parallel_retrieval_and_releases_only_reviewed_answer(monkeypatch):
    monkeypatch.setattr(graph_module, "get_safe_llm", lambda role: _TelemetryLLM(role))
    monkeypatch.setattr(graph_module, "PlannerAgent", _Planner)
    monkeypatch.setattr(graph_module, "ClarifierAgent", _Clarifier)
    monkeypatch.setattr(graph_module, "ReaderAgent", _Reader)
    monkeypatch.setattr(graph_module, "ResearcherAgent", _Researcher)
    monkeypatch.setattr(graph_module, "WriterAgent", _Writer)
    monkeypatch.setattr(graph_module, "ReviewerAgent", _Reviewer)

    result = graph_module.build_graph().invoke(AgentState(question="Find current papers"))

    assert result["reader_status"] == "skipped"
    assert result["researcher_status"] == "completed"
    assert result["evidence_available"] is True
    assert result["reviewed_answer"] == "Grounded draft [1]."
    assert result["workflow_status"] == "accepted"
    assert set(result["model_usage"]) == {
        "planner", "clarifier", "reader", "researcher", "writer", "reviewer"
    }
    assert result["model_usage"]["clarifier"]["status"] == "not_called"
    assert result["model_usage"]["reader"]["status"] == "not_applicable"
    assert result["model_usage"]["researcher"]["model"] == "test/researcher"
    assert result["model_usage"]["reviewer"]["model"] == "test/reviewer"
    assert result["writer_model"] == result["model_usage"]["writer"]
    json.dumps(result["model_usage"])


def test_graph_stops_before_writer_when_both_retrieval_branches_are_empty(monkeypatch):
    class EmptyResearcher(_Researcher):
        def run(self, state, progress_callback=None):
            state.researcher_status = "completed"
            state.external_context = []
            return state

    class FailingWriter(_Writer):
        def run(self, state):
            raise AssertionError("Writer must not run without evidence")

    monkeypatch.setattr(graph_module, "get_safe_llm", lambda _role: None)
    monkeypatch.setattr(graph_module, "PlannerAgent", _Planner)
    monkeypatch.setattr(graph_module, "ClarifierAgent", _Clarifier)
    monkeypatch.setattr(graph_module, "ReaderAgent", _Reader)
    monkeypatch.setattr(graph_module, "ResearcherAgent", EmptyResearcher)
    monkeypatch.setattr(graph_module, "WriterAgent", FailingWriter)
    monkeypatch.setattr(graph_module, "ReviewerAgent", _Reviewer)

    result = graph_module.build_graph().invoke(AgentState(question="no evidence"))

    assert result["workflow_status"] == "rejected"
    assert result["failure_code"] == "insufficient_grounded_evidence"
    assert result["model_usage"]["writer"]["status"] == "not_called"


def test_run_chat_workflow_invokes_compiled_graph(monkeypatch):
    calls = {}

    class _CompiledGraph:
        def invoke(self, state, config):
            calls["state"] = state
            calls["config"] = config
            return {
                "reviewed_answer": "Accepted graph result",
                "model_usage": {"writer": {"status": "selected"}},
                "writer_model": {"status": "selected"},
            }

    def fake_build_graph(*, checkpointer, progress_callback):
        calls["checkpointer"] = checkpointer
        calls["progress_callback"] = progress_callback
        return _CompiledGraph()

    checkpointer = object()
    callback = lambda _event: None
    monkeypatch.setattr(rag_workflow, "get_graph_checkpointer", lambda: checkpointer)
    monkeypatch.setattr(rag_workflow, "build_graph", fake_build_graph)

    result = rag_workflow.run_chat_workflow(
        "Verify LangGraph runtime",
        article_id=42,
        progress_callback=callback,
    )

    assert calls["checkpointer"] is checkpointer
    assert calls["progress_callback"] is callback
    assert calls["state"].question == "Verify LangGraph runtime"
    assert calls["state"].article_id == 42
    assert calls["config"]["configurable"]["thread_id"].startswith("research-")
    assert result["reviewed_answer"] == "Accepted graph result"
    assert result["writer_model"]["status"] == "selected"
