from langgraph.graph import END, StateGraph

from app.agents.clarifier import ClarifierAgent
from app.agents.planner import PlannerAgent
from app.agents.reader import ReaderAgent
from app.agents.researcher import ResearcherAgent
from app.agents.reviewer import ReviewerAgent
from app.agents.writer import WriterAgent
from app.workflows.states import AgentState


# TODO: Initialize agent instances
planner = PlannerAgent()
clarifier = ClarifierAgent()
researcher = ResearcherAgent()
reader = ReaderAgent()
writer = WriterAgent()
reviewer = ReviewerAgent()


def plan_node(state: AgentState):
    # TODO: Call planner logic
    return planner.plan(state)


def clarify_node(state: AgentState):
    # TODO: Call clarifier logic
    return clarifier.verify(state)


def read_node(state: AgentState):
    # TODO: Call reader logic
    return reader.read(state)


def research_node(state: AgentState):
    # TODO: Call researcher logic
    return researcher.search(state)


def write_node(state: AgentState):
    # TODO: Call writer logic
    return writer.write(state)


def review_node(state: AgentState):
    # TODO: Call reviewer logic
    return reviewer.review(state)


def route_after_planner(state: AgentState):
    if state.need_clarification:
        return "clarifier"
    return "reader"


def route_after_reader(state: AgentState):
    if state.need_external_search:
        return "researcher"
    return "writer"


def route_after_reviewer(state: AgentState):
    if state.confidence_score < 0.7 and state.iteration_count < state.max_iterations:
        return "rewrite"
    return "accept"


workflow = StateGraph(AgentState)
workflow.add_node("planner", plan_node)
workflow.add_node("clarifier", clarify_node)
workflow.add_node("reader", read_node)
workflow.add_node("researcher", research_node)
workflow.add_node("writer", write_node)
workflow.add_node("reviewer", review_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges("planner", route_after_planner, {"clarifier": "clarifier", "reader": "reader"})
workflow.add_edge("clarifier", "reader")
workflow.add_conditional_edges("reader", route_after_reader, {"researcher": "researcher", "writer": "writer"})
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_conditional_edges("reviewer", route_after_reviewer, {"rewrite": "writer", "accept": END})

compiled_workflow = workflow.compile()


def run_chat_workflow(question: str, article_id: int, session_id: str | None = None) -> AgentState:
    """
    TODO: Execute complete agentic workflow
    Input: question, article_id, optional session_id
    Output: AgenState with draft_answer/reviewed_answer and citations
    """
    state = AgentState(question=question, article_id=article_id, session_id=session_id)
    result = compiled_workflow.invoke(state.model_dump())
    if isinstance(result, AgentState):
        return result
    return AgentState(**result)
