from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import effective_question, log
from app.prompts.writer_prompt import WRITER_PROMPT
from app.tools.citation import format_apa_reference
from app.workflows.states import AgentState


class WriterAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        # Extract research notes from external_context
        notes = next(
            (r["content"] for r in state.external_context if r.get("title") == "__research_notes__"),
            "No external research available.",
        )

        # Build source index — exclude the internal research_notes entry
        raw_sources = [r for r in state.external_context if r.get("title") != "__research_notes__"]
        source_index_lines = []
        apa_references = []
        for i, r in enumerate(raw_sources):
            source_index_lines.append(f"[{i + 1}] {r.get('title', '')}— {r['url']}")
            apa_references.append(format_apa_reference(i + 1, r))

        source_index = "\n".join(source_index_lines)
        apa_ref_block = "\n".join(apa_references)

        vector_section = ""
        if state.vector_context:
            vector_section = "\n\n=== PDF / Vector Context (cite as [PDF-N]) ===\n" + "\n".join(
                f"[PDF-{i + 1}] Section: {c.get('section', '?')} | {c.get('content', '')[:400]}"
                for i, c in enumerate(state.vector_context)
            )

        feedback_section = ""
        if state.review_feedback and state.review_feedback not in ("No feedback.", None):
            feedback_section = (
                f"\n\n=== REVIEWER FEEDBACK — YOU MUST ADDRESS ALL POINTS BELOW ===\n"
                f"{state.review_feedback}\n"
                f"=== END FEEDBACK ==="
            )

        question = effective_question(state)
        context = f"""Research Question: {question}
Focus Sections: {', '.join(state.focus_sections) or 'All'}

=== Research Notes (extracted from sources) ===
{notes}

=== Source Index (for inline [N] citations) ===
{source_index}

=== Pre-formatted APA References (copy verbatim into References section) ===
{apa_ref_block}
{vector_section}
{feedback_section}
"""

        prompt = WRITER_PROMPT.replace("{QUESTION}", question)
        res = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=context)])
        state.draft_answer = res.content
        log(state, f"\n[WriterAgent] Draft written — iteration {state.iteration_count + 1} ({len(res.content)} chars)")
        return state
