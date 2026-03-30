from app.workflows.states import AgentState
from app.tools.external_search import external_search


class ResearcherAgent:
    """Search external sources (Google CSE and Semantic Scholar)."""

    def search(self, state: AgentState) -> AgentState:
        if not state.need_external_search:
            state.external_context = []
            return state

        state.external_context = external_search(state.question, top_k=5)
        return state
