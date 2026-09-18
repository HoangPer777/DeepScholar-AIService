from langchain_core.messages import HumanMessage, SystemMessage

from app.core.safe_llm import AllLLMProvidersFailed
from app.core.utils import effective_question, log
from app.prompts.writer_prompt import WRITER_PROMPT
from app.tools.citation import (
    canonicalize_references,
    format_apa_reference,
    select_citable_sources,
)
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

        # Use one bounded evidence-bearing set for indexes, excerpts, and
        # canonical references. Public indexes remain aligned with API sources.
        citable_sources = select_citable_sources(state.external_context)
        source_index_lines = []
        references_by_index = {}
        for source_index_number, source in citable_sources:
            source_index_lines.append(
                f"[{source_index_number}] {source.get('title', '')}— {source.get('url', '')}"
            )
            references_by_index[source_index_number] = format_apa_reference(
                source_index_number, source
            )

        source_index = "\n".join(source_index_lines) or "No citable external sources."
        apa_ref_block = "\n".join(references_by_index.values())
        allowed_citations = ", ".join(
            f"[{source_index_number}]" for source_index_number, _ in citable_sources
        ) or "None"

        # Research notes are LLM-generated and may compress or misstate a
        # source. Give Writer citation-bound, verbatim evidence so claims can
        # be checked against the discovered documents instead of model memory.
        evidence_lines = []
        for source_index_number, source in citable_sources:
            content = (source.get("content") or "").strip()
            evidence_lines.append(
                f"[{source_index_number}] {source.get('title', 'Untitled')}\n"
                f"Citation-bound verbatim excerpt: {content[:1200]}"
            )
        source_evidence = "\n\n".join(evidence_lines) or "No citation-bound external excerpts available."

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

Allowed external citation markers: {allowed_citations}
Do not use any external [N] marker not listed above.

=== Citation-Bound Source Evidence (authoritative for factual claims) ===
{source_evidence}

=== Pre-formatted APA References (copy verbatim into References section) ===
{apa_ref_block}
{vector_section}
{feedback_section}
"""

        prompt = WRITER_PROMPT.replace("{QUESTION}", question)
        try:
            messages = [SystemMessage(content=prompt), HumanMessage(content=context)]
            for attempt in range(2):
                res = self.llm.invoke(messages)
                draft = str(getattr(res, "content", "") or "").strip()
                if draft:
                    state.draft_answer = draft
                    break
                if attempt == 0:
                    log(state, "[WriterAgent] Empty model response; retrying once with the same grounded evidence")
            else:
                state.draft_answer = None
                state.failure_code = "writer_empty_response"
                state.failure_message = "Writer model returned an empty response after one retry."
                state.workflow_status = "failed"
                log(state, "[WriterAgent] FAILED — model returned empty responses")
                return state
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
        if state.draft_answer:
            state.draft_answer = canonicalize_references(
                state.draft_answer, references_by_index
            )
        log(state, f"\n[WriterAgent] Draft written — iteration {state.iteration_count + 1} ({len(state.draft_answer or '')} chars)")
        return state
