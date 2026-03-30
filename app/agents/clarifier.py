from app.workflows.states import AgentState


class ClarifierAgent:
    """Clarify ambiguous questions before heavy retrieval begins."""

    def verify(self, state: AgentState) -> AgentState:
        if not state.need_clarification:
            state.clarification_question = None
            return state

        q = (state.question or "").strip()
        state.clarification_question = (
            "Cau hoi cua ban con mo ho. Vui long bo sung pham vi (nam, dataset, phuong phap hoac paper cu the) "
            f"de minh deep research chinh xac hon. Cau hien tai: '{q}'."
        )
        return state
