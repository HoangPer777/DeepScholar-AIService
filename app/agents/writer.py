from langchain_core.messages import HumanMessage, SystemMessage

from app.core.safe_llm import AllLLMProvidersFailed
from app.core.utils import effective_question, log
from app.prompts.writer_prompt import WRITER_PROMPT
from app.tools.citation import format_apa_reference
from app.workflows.states import AgentState


class WriterAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        raw_sources = [r for r in state.external_context if r.get("title") != "__research_notes__"]
        if state.workflow_status == "evidence_ready" and not raw_sources and not state.vector_context:
            state.failure_code = "insufficient_grounded_evidence"
            state.failure_message = "No internal or external evidence was available for a grounded report."
            state.workflow_status = "failed"
            state.draft_answer = None
            log(state, "[WriterAgent] SKIPPED — no grounded evidence")
            return state
        # Extract research notes from external_context
        notes = next(
            (r["content"] for r in state.external_context if r.get("title") == "__research_notes__"),
            "No external research available.",
        )

        # Build source index — exclude the internal research_notes entry
        source_index_lines = []
        apa_references = []
        for i, r in enumerate(raw_sources):
            source_index_lines.append(f"[{i + 1}] {r.get('title', '')}— {r['url']}")
            apa_references.append(format_apa_reference(i + 1, r))

        source_index = "\n".join(source_index_lines)
        apa_ref_block = "\n".join(apa_references)

        # Research notes are LLM-generated and may compress or misstate a
        # source. Give Writer bounded verbatim evidence so claims can be
        # checked against the discovered documents instead of model memory.
        evidence_lines = []
        for i, source in enumerate(raw_sources[:12]):
            content = (source.get("content") or "").strip()
            if not content:
                continue
            evidence_lines.append(
                f"[{i + 1}] {source.get('title', 'Untitled')}\n"
                f"Verbatim excerpt: {content[:1200]}"
            )
        source_evidence = "\n\n".join(evidence_lines) or "No verbatim external excerpts available."

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

=== Verbatim Source Evidence (authoritative for factual claims) ===
{source_evidence}

=== Pre-formatted APA References (copy verbatim into References section) ===
{apa_ref_block}
{vector_section}
{feedback_section}
"""

        prompt = WRITER_PROMPT.replace("{QUESTION}", question)
        try:
            res = self.llm.invoke([SystemMessage(content=prompt), HumanMessage(content=context)])
            state.draft_answer = res.content
        except AllLLMProvidersFailed as exc:
            log(state, f"[WriterAgent] LLM unavailable, using source-based fallback: {exc}")
            references = apa_ref_block or source_index or "No sources available."
            state.draft_answer = (
                f"# Source-based research summary\n\n"
                f"OpenRouter free models are currently unavailable or rate-limited, "
                f"so this fallback answer is generated directly from collected sources.\n\n"
                f"## Question\n{question}\n\n"
                f"## Key notes\n{notes}\n\n"
                f"## References\n{references}"
            )
        log(state, f"\n[WriterAgent] Draft written — iteration {state.iteration_count + 1} ({len(state.draft_answer)} chars)")
        return state
