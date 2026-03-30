from app.workflows.states import AgentState
from app.tools.vector_search import vector_search


class ReaderAgent:
    """Read relevant chunks from vector DB."""

    def read(self, state: AgentState) -> AgentState:
        results = vector_search(state.question, top_k=7)

        if state.focus_sections:
            focus_words = {s.lower() for s in state.focus_sections}
            filtered = []
            for item in results:
                text = f"{item.get('title', '')} {item.get('text', '')}".lower()
                if any(section in text for section in focus_words):
                    filtered.append(item)
            state.vector_context = filtered if filtered else results
        else:
            state.vector_context = results

        return state
