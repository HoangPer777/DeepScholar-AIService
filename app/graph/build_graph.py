from langgraph.graph import END, START, StateGraph

from app.agents.clarifier import ClarifierAgent
from app.agents.planner import PlannerAgent
from app.agents.reader import ReaderAgent
from app.agents.researcher import ResearcherAgent
from app.agents.reviewer import ReviewerAgent
from app.agents.writer import WriterAgent
from app.core.llm import get_safe_llm
from app.workflows.states import AgentState


def _review_router(state: AgentState) -> str:
    if (
        state.reviewed_answer is None
        and state.confidence_score < 0.7
        and state.iteration_count < state.max_iterations
    ):
        return "writer"
    return END


def build_graph():
    # V12: Per-agent LLM via SafeLLM router (falls back to Groq if no OPENROUTER_API_KEY)
    planner    = PlannerAgent(get_safe_llm("planner"))
    clarifier  = ClarifierAgent(get_safe_llm("clarifier"))
    researcher = ResearcherAgent(get_safe_llm("researcher"))
    reader     = ReaderAgent()       # No LLM needed — pure vector search
    writer     = WriterAgent(get_safe_llm("writer"))
    reviewer   = ReviewerAgent(get_safe_llm("reviewer"))

    graph = StateGraph(AgentState)

    graph.add_node("planner",    planner.run)
    graph.add_node("clarifier",  clarifier.run)
    graph.add_node("researcher", researcher.run)
    graph.add_node("reader",     reader.run)
    graph.add_node("writer",     writer.run)
    graph.add_node("reviewer",   reviewer.run)

    graph.add_edge(START,        "planner")
    graph.add_edge("planner",    "clarifier")
    graph.add_edge("clarifier",  "researcher")
    graph.add_edge("researcher", "reader")
    graph.add_edge("reader",     "writer")
    graph.add_edge("writer",     "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        _review_router,
        {"writer": "writer", END: END},
    )

    return graph.compile()
