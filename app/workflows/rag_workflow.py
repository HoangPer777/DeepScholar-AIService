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
    # TODO: Route to Clarifier or Reader based on planning decision
    pass


def route_after_reader(state: AgentState):
    # TODO: Route to Researcher or Writer based on need_external_search
    pass


def route_after_reviewer(state: AgentState):
    # TODO: Route to Writer (rewrite) or END based on confidence score and iterations
    pass


# TODO: Build LangGraph workflow
workflow = StateGraph(AgentState)

# TODO: Add all nodes
# TODO: Set entry point
# TODO: Connect nodes with edges and conditional edges
# TODO: Compile workflow

compiled_workflow = None


def run_chat_workflow(question: str, article_id: int, session_id: str | None = None) -> AgentState:
    """
    TODO: Execute complete agentic workflow
    Input: question, article_id, optional session_id
    Output: AgenState with draft_answer/reviewed_answer and citations
    """
    state = AgentState(question=question, article_id=article_id, session_id=session_id)
    # TODO: Invoke compiled workflow
    # result = compiled_workflow.invoke(state)
    # TODO: Return AgentState
    return state
