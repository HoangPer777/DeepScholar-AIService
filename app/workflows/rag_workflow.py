from app.graph.build_graph import build_graph
from app.workflows.states import AgentState

_compiled = build_graph()


def run_chat_workflow(
    question: str,
    article_id: int | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Execute the full agentic workflow.
    article_id=None → ReaderAgent skips PGVector (deep research mode).
    Returns the final state as a dict.
    """
    state = AgentState(question=question, article_id=article_id)
    result = _compiled.invoke(state)
    return result
