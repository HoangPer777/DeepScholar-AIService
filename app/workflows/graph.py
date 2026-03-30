from langgraph.graph import END, START, StateGraph

from app.agents.agents import (
    clarifier,
    dispatch_node,
    dispatch_router,
    memory_agent,
    planner,
    planner_router,
    ranking,
    reader,
    researcher,
    reviewer,
    review_router,
    writer,
)
from app.core.state import AgentState


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("clarifier", clarifier)
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("reader", reader)
    graph.add_node("researcher", researcher)
    graph.add_node("memory_agent", memory_agent)
    graph.add_node("ranking", ranking)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", planner_router, ["clarifier", "dispatch"])
    graph.add_edge("clarifier", "dispatch")
    graph.add_conditional_edges("dispatch", dispatch_router, ["reader", "researcher", "memory_agent"])

    graph.add_edge("reader", "ranking")
    graph.add_edge("researcher", "ranking")
    graph.add_edge("memory_agent", "ranking")

    graph.add_edge("ranking", "writer")
    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        review_router,
        {
            "accept": END,
            "rewrite": "writer",
        },
    )

    return graph.compile()


app = build_graph()
