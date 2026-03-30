from app.workflows.states import AgentState


class ReviewerAgent:
    """Review generated answer and assign confidence score."""

    def review(self, state: AgentState) -> AgentState:
        state.iteration_count += 1

        draft = (state.draft_answer or "").strip()
        has_answer = bool(draft)
        has_context = len(state.vector_context) + len(state.external_context) > 0
        has_citations = len(state.citations) > 0 and "[1]" in draft

        score = 0.0
        if has_answer:
            score += 0.35
        if has_context:
            score += 0.30
        if has_citations:
            score += 0.35

        state.confidence_score = min(score, 1.0)

        if state.confidence_score < 0.7 and state.iteration_count < state.max_iterations:
            if not has_answer:
                state.review_feedback = "Answer is empty. You must provide a direct response."
            elif not has_citations:
                state.review_feedback = "Missing inline citation markers like [1], [2]."
            else:
                state.review_feedback = "Improve clarity and keep strong grounding in retrieved evidence."
            state.reviewed_answer = None
            return state

        state.review_feedback = None
        state.reviewed_answer = state.draft_answer
        return state
