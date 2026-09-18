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


_EVIDENCE_NOT_FOUND = {
    "evidence not found in sources.",
    "evidence not found in sources",
}


def _has_usable_evidence(citable_sources, vector_context) -> bool:
    return bool(citable_sources or any((item.get("content") or "").strip() for item in vector_context))


def _build_source_based_draft(
    question: str,
    citable_sources: list[tuple[int, dict]],
    references_by_index: dict[int, str],
    vector_context: list[dict],
) -> str:
    """Build an honest bounded draft when the model returns the empty sentinel."""
    evidence = []
    for index, source in citable_sources:
        content = (source.get("content") or "").strip()[:900]
        if content:
            evidence.append(f"The supplied excerpt for source [{index}] states: {content} [{index}]")
    for index, chunk in enumerate(vector_context, start=1):
        content = (chunk.get("content") or "").strip()[:500]
        if content:
            evidence.append(f"The supplied internal excerpt [PDF-{index}] states: {content} [PDF-{index}]")

    evidence_text = "\n\n".join(evidence) or "No usable evidence excerpt was returned."
    comparison = (
        "The available excerpts are presented side by side below; no unsupported performance comparison is inferred. "
        + " ".join(f"[{index}]" for index, _ in citable_sources[:2])
        if len(citable_sources) >= 2
        else "The supplied excerpts do not support a direct quantitative comparison."
    )
    references = "\n".join(references_by_index.values())
    return (
        f"## Abstract\nThis source-grounded draft addresses: {question}. "
        f"It reports only the evidence supplied below.\n\n"
        f"## Introduction\nThe question is examined using the retrieved source excerpts. "
        f"No claim beyond those excerpts is added.\n\n"
        f"## Methodology\n{evidence_text}\n\n"
        f"## Results & Key Advances\n{evidence_text}\n\n"
        f"## Discussion\n{comparison}\n\n"
        f"## Conclusion\nThe available evidence is limited to the cited excerpts above. "
        f"Additional source text is required for stronger methodological or quantitative conclusions.\n\n"
        f"## References\n{references}"
    )


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
                if draft.casefold() in _EVIDENCE_NOT_FOUND:
                    if attempt == 0:
                        log(
                            state,
                            "[WriterAgent] Model returned the evidence sentinel despite retrieved context; retrying",
                        )
                        continue
                    if _has_usable_evidence(citable_sources, state.vector_context):
                        state.draft_answer = _build_source_based_draft(
                            question,
                            citable_sources,
                            references_by_index,
                            state.vector_context,
                        )
                        log(state, "[WriterAgent] Used bounded source-based fallback after repeated evidence sentinel")
                        break
                    state.draft_answer = None
                    state.failure_code = "writer_insufficient_evidence"
                    state.failure_message = "Writer could not find usable evidence excerpts for this question."
                    state.workflow_status = "failed"
                    log(state, "[WriterAgent] FAILED — no usable evidence excerpts")
                    return state
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
