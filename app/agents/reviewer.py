from app.workflows.states import AgentState


class ReviewerAgent:
    """TODO: Review and score quality of generated answer"""

    def review(self, state: AgentState) -> AgentState:
        """
        TODO: Implement review logic:
        1. Check if answer has sufficient context
        2. Verify citations are included
        3. Check for hallucinations
        4. Score confidence (0.0-1.0)
        5. If score < 0.7 and iterations < max, request rewrite
        6. Otherwise approve and set reviewed_answer
        """
        # TODO: Implementation
        return state
