from app.workflows.states import AgentState


class WriterAgent:
    """Synthesize context and write answer with citations."""

    def write(self, state: AgentState) -> AgentState:
        if state.need_clarification and not state.vector_context and not state.external_context:
            state.draft_answer = state.clarification_question or "Please clarify your question so I can continue."
            state.citations = []
            return state

        combined = state.vector_context + state.external_context
        if not combined:
            state.draft_answer = (
                "Khong tim thay ngu canh du tin cay de tra loi chinh xac. "
                "Vui long cung cap them pham vi (nam, domain, paper cu the) de minh deep research lai."
            )
            state.citations = []
            return state

        selected = combined[: min(len(combined), 6)]
        citations = []
        evidence_lines = []
        for idx, ctx in enumerate(selected, 1):
            title = ctx.get("title") or f"Source {idx}"
            source = ctx.get("source", "unknown")
            snippet = (ctx.get("text") or "").replace("\n", " ").strip()
            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "..."
            evidence_lines.append(f"[{idx}] {title} ({source}): {snippet}")
            citations.append(
                {
                    "index": idx,
                    "title": title,
                    "source": source,
                    "url": ctx.get("url", ""),
                    "doi": ctx.get("doi", ""),
                }
            )

        feedback_prefix = ""
        if state.review_feedback and state.iteration_count > 0:
            feedback_prefix = f"Da rewrite theo feedback reviewer: {state.review_feedback}\n\n"

        summary = (
            f"{feedback_prefix}Tra loi cho cau hoi: {state.question}\n\n"
            "Tong hop tu ngu canh truy xuat duoc:\n"
            + "\n".join(evidence_lines)
            + "\n\nKet luan: thong tin tren da duoc tong hop tu PDF context va external sources (neu co)."
        )

        state.draft_answer = summary
        state.citations = citations
        return state
