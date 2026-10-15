from unittest.mock import MagicMock, patch

import fakeredis

from app.agents.fast_chat import FastChatAgent
from app.core.memory_store import MemoryStore
from app.schemas.chat_models import ContextWindow, ResearchReport
from app.workflows.rag_workflow import _research_context_content, run_chat_workflow


def _rejected_result():
    return {
        "reviewed_answer": None,
        "draft_answer": "An evidence-based but unreviewed draft.",
        "review_decision": "rejected",
        "failure_code": "research_review_rejected",
        "confidence_score": 0.62,
        "review_feedback": "Remove unsupported claims.",
        "external_context": [],
    }


def test_unreviewed_draft_is_eligible_but_reviewer_failure_is_not():
    selected = _research_context_content(_rejected_result())
    assert selected == ("An evidence-based but unreviewed draft.", "unreviewed")

    failed = _rejected_result()
    failed["failure_code"] = "reviewer_unavailable"
    assert _research_context_content(failed) is None


def test_rejected_draft_is_persisted_and_fast_chat_ready_is_reported():
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server)
    graph = MagicMock()
    graph.invoke.return_value = _rejected_result()

    with (
        patch("app.workflows.rag_workflow.create_redis_client", return_value=client),
        patch("app.workflows.rag_workflow.get_graph_checkpointer", return_value=None),
        patch("app.workflows.rag_workflow.build_graph", return_value=graph),
    ):
        result = run_chat_workflow("What are the key findings?", session_id="test-unreviewed")

    context = MemoryStore(client).get_context_window("test-unreviewed")
    assert result["fast_chat_ready"] is True
    assert context.research_report.answer == "An evidence-based but unreviewed draft."
    assert context.research_report.review_status == "unreviewed"
    assert context.research_report.review_feedback == "Remove unsupported claims."
    client.close()


def test_fast_chat_prompt_can_use_unreviewed_report_without_claiming_review():
    llm = MagicMock()
    llm.invoke.return_value.content = "The report says the main finding is improved retrieval."
    context = ContextWindow(
        session_id="test-unreviewed",
        research_report=ResearchReport(
            answer="An evidence-based but unreviewed draft.",
            review_status="unreviewed",
            review_feedback="Remove unsupported claims.",
        ),
    )

    result = FastChatAgent(llm).run("What are the key findings?", context)

    prompt = llm.invoke.call_args.args[0][1].content
    assert "REPORT REVIEW STATUS: UNREVIEWED" in prompt
    assert "Remove unsupported claims." in prompt
    assert result["answer"].startswith("The report says")
    assert result["research_review_status"] == "unreviewed"
    assert result["is_fast_chat"] is True
