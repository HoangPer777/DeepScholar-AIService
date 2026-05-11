"""
ReviewerAgent — V13.

V13 changes:
- Source quality gate: check academic_ratio, citation quality, low-quality domains
- Reviewer rejects/rewrites if academic_ratio < 0.3 regardless of LLM score
- Score capped at 0.60 when academic_ratio < 0.3
- Source list included in LLM input for quality-aware evaluation
"""
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import effective_question, log, safe_json
from app.prompts.reviewer_prompt import REVIEWER_PROMPT
from app.tools.source_filter import LOW_QUALITY_DOMAINS
from app.workflows.states import AgentState


# Academic source types — V13 includes openalex and crossref
_ACADEMIC_SOURCE_TYPES = {"arxiv", "semantic_scholar", "alphaxiv", "openalex", "crossref"}

# Score cap when academic_ratio < 0.3
_LOW_ACADEMIC_SCORE_CAP = 0.60


def _source_quality_gate(
    sources: List[Dict],
    need_external_search: bool,
) -> Tuple[bool, List[str]]:
    """
    Kiểm tra source quality trước khi LLM evaluation.

    Preconditions:
    - sources là list of dicts (external_context, có thể chứa __research_notes__)
    - need_external_search là bool

    Postconditions:
    - Nếu need_external_search == False: return (True, []) — gate skipped
    - Trả về (passed: bool, failed_criteria: List[str])
    - Không raise exception

    Criteria:
    (a) academic_ratio >= 0.3 (academic_count / total_count)
    (b) at least 1 source with citation_count > 10
        (relaxed: skip if academic_ratio >= 0.5)
    (c) no source URL from LOW_QUALITY_DOMAINS

    Args:
        sources: List of source dicts from external_context
        need_external_search: Whether external search was needed

    Returns:
        Tuple of (gate_passed, list_of_failed_criteria)
    """
    # Gate is skipped when external search was not needed
    if not need_external_search:
        return True, []

    # Filter out internal research notes
    real_sources = [s for s in sources if s.get("title") != "__research_notes__"]
    total = len(real_sources)

    if total == 0:
        # No sources at all when external search was needed — critical failure
        return False, ["no_sources_found"]

    academic = [s for s in real_sources if s.get("source_type") in _ACADEMIC_SOURCE_TYPES]
    academic_count = len(academic)
    academic_ratio = academic_count / total

    failed: List[str] = []

    # Criterion (a): academic_ratio >= 0.3
    if academic_ratio < 0.3:
        failed.append("insufficient_academic_sources")

    # Criterion (b): at least 1 source with citation_count > 10
    # Relaxed: skip if academic_ratio >= 0.5
    if academic_ratio < 0.5:
        has_cited = any(s.get("citation_count", 0) > 10 for s in real_sources)
        if not has_cited:
            failed.append("no_highly_cited_source")

    # Criterion (c): no LOW_QUALITY_DOMAINS in sources
    has_low_quality = any(
        any(domain in (s.get("url") or "").lower() for domain in LOW_QUALITY_DOMAINS)
        for s in real_sources
    )
    if has_low_quality:
        failed.append("low_quality_domain_present")

    return len(failed) == 0, failed


def _build_source_summary(sources: List[Dict]) -> str:
    """Build a compact source list string for LLM input."""
    real_sources = [s for s in sources if s.get("title") != "__research_notes__"]
    if not real_sources:
        return "No external sources."
    lines = [
        f"[{i + 1}] {s.get('title', 'Untitled')} "
        f"({s.get('source_type', 'web')}, citations: {s.get('citation_count', 0)})"
        for i, s in enumerate(real_sources)
    ]
    return "\n".join(lines)


class ReviewerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        # V13: Run source quality gate before LLM evaluation
        gate_passed, gate_failed = _source_quality_gate(
            state.external_context,
            state.need_external_search,
        )

        # Compute metrics for logging
        real_sources = [s for s in state.external_context if s.get("title") != "__research_notes__"]
        total_count = len(real_sources)
        academic_count = sum(1 for s in real_sources if s.get("source_type") in _ACADEMIC_SOURCE_TYPES)
        academic_ratio = academic_count / max(total_count, 1)

        log(state, f"\n[ReviewerAgent] Source quality: {academic_count}/{total_count} academic (ratio={academic_ratio:.2f})")
        if gate_failed:
            log(state, f"  [ReviewerAgent] Gate failed: {gate_failed}")

        # Critical: no sources at all when external search was needed
        if "no_sources_found" in gate_failed and state.need_external_search:
            state.confidence_score = 0.0
            state.review_feedback = "No external sources found. Cannot evaluate grounding quality."
            state.iteration_count += 1
            log(state, f"\n[ReviewerAgent] Iteration {state.iteration_count} — REWRITE (no sources)")
            if state.iteration_count >= state.max_iterations:
                state.reviewed_answer = state.draft_answer
                log(state, "  -> MAX ITERATIONS reached — using best draft")
            return state

        # V13: Include source list in LLM input for quality-aware evaluation
        source_summary = _build_source_summary(state.external_context)

        res = self.llm.invoke([
            SystemMessage(content=REVIEWER_PROMPT),
            HumanMessage(content=(
                f"Research Question: {effective_question(state)}\n\n"
                f"=== Sources Used ===\n{source_summary}\n\n"
                f"=== Draft ===\n{state.draft_answer}"
            )),
        ])

        raw = res.content.strip()
        data = safe_json(raw)
        if not data:
            log(state, f"  [WARN] Reviewer JSON parse failed:\n{raw[:400]}")
            data = {
                "score":           0.55,
                "decision":        "rewrite",
                "failed_criteria": ["reviewer parse error"],
                "feedback":        raw[:400],
            }

        score = float(data.get("score", 0.55))
        decision = data.get("decision", "rewrite")
        failed = list(data.get("failed_criteria", []))
        feedback = data.get("feedback", "No feedback.")

        # V13: Merge gate failures into LLM failed_criteria
        for gf in gate_failed:
            if gf not in failed:
                failed.append(gf)

        # V13: Cap score if academic_ratio < 0.3
        if academic_ratio < 0.3 and score > _LOW_ACADEMIC_SCORE_CAP:
            log(state, f"  [ReviewerAgent] Score capped: {score:.2f} → {_LOW_ACADEMIC_SCORE_CAP} (academic_ratio={academic_ratio:.2f})")
            score = _LOW_ACADEMIC_SCORE_CAP

        # V13: Gate failure overrides LLM decision
        if gate_failed:
            decision = "rewrite"

        # Hard rule: critical failures override LLM decision
        critical = {"comparison", "explanation", "hallucination", "methodology prose", "repeated content"}
        if failed and any(any(c in f.lower() for c in critical) for f in failed):
            decision = "rewrite"
            if score >= 0.7:
                score = min(score, 0.68)

        state.confidence_score = score
        state.review_feedback = feedback
        state.iteration_count += 1

        log(state, f"\n[ReviewerAgent] Iteration {state.iteration_count}")
        log(state, f"  score           : {state.confidence_score:.2f}")
        log(state, f"  decision        : {decision}")
        log(state, f"  failed_criteria : {failed}")
        log(state, f"  feedback        : {feedback[:200]}")

        if decision == "accept" and state.confidence_score >= 0.7:
            state.reviewed_answer = state.draft_answer
            log(state, "  -> ACCEPTED")
        else:
            log(state, f"  -> REWRITE ({state.iteration_count}/{state.max_iterations})")
            if state.iteration_count >= state.max_iterations:
                state.reviewed_answer = state.draft_answer
                log(state, "  -> MAX ITERATIONS reached — using best draft")

        return state
