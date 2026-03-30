from app.workflows.states import AgentState


class PlannerAgent:
    """Analyze question and determine route decisions for the graph."""

    def plan(self, state: AgentState) -> AgentState:
        question = (state.question or "").strip()
        lowered = question.lower()

        ambiguous_tokens = [
            "giải thích",
            "explain",
            "cái này",
            "this",
            "nó",
            "something",
            "help",
        ]
        external_tokens = [
            "latest",
            "new",
            "2024",
            "2025",
            "2026",
            "benchmark",
            "state-of-the-art",
            "sota",
            "compare",
            "so sánh",
        ]

        state.need_clarification = len(question.split()) < 4 or any(token in lowered for token in ambiguous_tokens)
        state.need_external_search = any(token in lowered for token in external_tokens)

        focus_sections = []
        section_mapping = {
            "method": "Method",
            "phương pháp": "Method",
            "result": "Results",
            "kết quả": "Results",
            "conclusion": "Conclusion",
            "kết luận": "Conclusion",
            "abstract": "Abstract",
            "introduction": "Introduction",
            "giới thiệu": "Introduction",
        }
        for token, section in section_mapping.items():
            if token in lowered and section not in focus_sections:
                focus_sections.append(section)
        if not focus_sections:
            focus_sections = ["Method", "Results", "Conclusion"]

        state.focus_sections = focus_sections
        return state
