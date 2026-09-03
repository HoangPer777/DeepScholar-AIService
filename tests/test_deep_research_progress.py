"""Targeted tests for structured Deep Research progress reporting."""

from __future__ import annotations

from unittest.mock import patch

from app.api import research
from app.workflows import rag_workflow


def test_initial_job_snapshot_is_backward_compatible():
    snapshot = research._initial_job_snapshot(debug=False)

    assert snapshot["status"] == "pending"
    assert snapshot["progress"]["phase"] == "queued"
    assert snapshot["activities"][0]["sequence"] == 1
    assert snapshot["activities"][0]["state"] == "completed"
    assert snapshot["source_previews"] == []


def test_progress_emitter_bounds_and_sanitizes_payload():
    snapshot = research._initial_job_snapshot()

    with patch.object(research._job_store, "update_job") as update_job:
        emitter = research._ProgressEmitter("task-progress", snapshot)
        for index in range(61):
            emitter.emit(
                {
                    "phase": "searching",
                    "state": "completed",
                    "title": f"Query {index}",
                    "detail": "x" * 700,
                    "metadata": {"completed_queries": index + 1},
                    "source_previews": [
                        {
                            "title": f"Source {index}",
                            "url": f"https://example.com/paper-{index}",
                            "source_type": "web",
                            "year": 2026,
                        },
                        {
                            "title": "Duplicate",
                            "url": "https://example.com/paper-0",
                            "source_type": "web",
                        },
                        {"title": "Unsafe", "url": "javascript:alert(1)"},
                    ],
                }
            )

    result = emitter.snapshot
    assert len(result["activities"]) == 50
    assert len(result["activities"][-1]["detail"]) == 500
    assert result["activities"][-1]["sequence"] == 62
    assert len(result["source_previews"]) == 20
    assert len({item["url"] for item in result["source_previews"]}) == 20
    assert all(item["url"].startswith("https://") for item in result["source_previews"])
    assert update_job.call_count == 61


def test_progress_emitter_failure_does_not_escape():
    with patch.object(
        research._job_store,
        "update_job",
        side_effect=RuntimeError("Redis unavailable"),
    ):
        emitter = research._ProgressEmitter(
            "task-resilient", research._initial_job_snapshot()
        )
        emitter.emit(
            {
                "phase": "planning",
                "state": "active",
                "title": "Planning",
            }
        )

    assert emitter.snapshot["progress"]["phase"] == "planning"


def test_unknown_phase_is_ignored():
    with patch.object(research._job_store, "update_job") as update_job:
        emitter = research._ProgressEmitter(
            "task-unknown", research._initial_job_snapshot()
        )
        emitter.emit({"phase": "internal_reasoning", "title": "Hidden"})

    assert emitter.snapshot["progress"]["phase"] == "queued"
    update_job.assert_not_called()


def test_researcher_parallel_search_reports_each_query(monkeypatch):
    events = []

    def fake_search(query):
        slug = query.replace(" ", "-")
        return [
            {
                "title": f"Paper for {query}",
                "url": f"https://arxiv.org/abs/{slug}",
                "source_type": "arxiv",
                "year": 2026,
            }
        ]

    monkeypatch.setattr("app.agents.researcher.academic_search", fake_search)
    from app.agents.researcher import _collect_sources_parallel

    results = _collect_sources_parallel(["query one", "query two"], events.append)

    assert len(results) == 2
    assert len(events) == 2
    assert {event["metadata"]["query"] for event in events} == {
        "query one",
        "query two",
    }
    assert sorted(event["metadata"]["completed_queries"] for event in events) == [1, 2]
    assert all(event["source_previews"] for event in events)


def test_full_workflow_emits_review_rewrite_and_completion():
    events = []

    class FakePlanner:
        def __init__(self, _llm):
            pass

        def run(self, state):
            state.need_external_search = True
            state.search_queries = ["academic query"]
            state.focus_sections = ["Methodology"]
            return state

    class FakeClarifier:
        def __init__(self, _llm):
            pass

        def run(self, state):
            return state

    class FakeResearcher:
        def __init__(self, _llm):
            pass

        def run(self, state, progress_callback=None):
            if progress_callback:
                progress_callback(
                    {
                        "phase": "synthesizing",
                        "state": "completed",
                        "title": "Sources ready",
                    }
                )
            state.external_context = [
                {
                    "title": "Paper",
                    "url": "https://arxiv.org/abs/test",
                    "source_type": "arxiv",
                }
            ]
            return state

    class FakeReader:
        def run(self, state):
            return state

    class FakeWriter:
        def __init__(self, _llm):
            pass

        def run(self, state):
            state.draft_answer = f"Draft {state.iteration_count + 1}"
            return state

    class FakeReviewer:
        def __init__(self, _llm):
            pass

        def run(self, state):
            state.iteration_count += 1
            if state.iteration_count == 1:
                state.confidence_score = 0.5
                state.review_feedback = "Add stronger evidence."
            else:
                state.confidence_score = 0.9
                state.review_feedback = "Accepted."
                state.reviewed_answer = state.draft_answer
            return state

    with (
        patch.object(rag_workflow, "get_safe_llm", return_value=object()),
        patch.object(rag_workflow, "PlannerAgent", FakePlanner),
        patch.object(rag_workflow, "ClarifierAgent", FakeClarifier),
        patch.object(rag_workflow, "ResearcherAgent", FakeResearcher),
        patch.object(rag_workflow, "ReaderAgent", FakeReader),
        patch.object(rag_workflow, "WriterAgent", FakeWriter),
        patch.object(rag_workflow, "ReviewerAgent", FakeReviewer),
    ):
        result = rag_workflow.run_chat_workflow(
            "Explain hybrid retrieval",
            progress_callback=events.append,
        )

    phases = [event["phase"] for event in events]
    assert "planning" in phases
    assert "searching" in phases
    assert "drafting" in phases
    assert "reviewing" in phases
    assert "rewriting" in phases
    assert phases[-2:] == ["finalizing", "completed"]
    assert result["reviewed_answer"] == "Draft 2"

    review_events = [
        event
        for event in events
        if event["phase"] == "reviewing" and event["state"] == "completed"
    ]
    assert review_events[0]["metadata"]["decision"] == "rewrite"
    assert review_events[1]["metadata"]["decision"] == "accept"


def test_workflow_progress_callback_failure_is_non_fatal():
    def broken_callback(_event):
        raise RuntimeError("UI observer failed")

    rag_workflow._emit_progress(
        broken_callback,
        phase="planning",
        state="active",
        title="Planning",
    )
