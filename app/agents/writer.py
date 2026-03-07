from app.workflows.states import AgentState


class WriterAgent:
    """TODO: Synthesize context and write answer with citations"""

    def write(self, state: AgentState) -> AgentState:
        """
        TODO: Implement answer writing:
        1. Check if need_clarification, return clarification_question
        2. Synthesize vector_context + external_context
        3. Generate draft_answer using LLM or template
        4. Build citations from sources
        5. Handle multi-turn rewrites with review_feedback
        """
        # TODO: Implementation
        return state
