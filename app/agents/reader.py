from app.workflows.states import AgentState


class ReaderAgent:
    """TODO: Read and retrieve relevant chunks from article via vector DB"""

    def read(self, state: AgentState) -> AgentState:
        """
        TODO: Implement vector DB query:
        1. Embed question
        2. Similarity search in PGVector
        3. Filter by focus_sections
        4. Return top-N chunks
        """
        # TODO: Implementation
        return state
