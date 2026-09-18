"""
ReviewerAgent — V13.

V13 changes:
- Source quality gate: check academic_ratio, citation quality, low-quality domains
- Reviewer rejects/rewrites if academic_ratio < 0.3 regardless of LLM score
- Score capped at 0.60 when academic_ratio < 0.3
- Source list included in LLM input for quality-aware evaluation
"""
import re
from typing import Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.safe_llm import AllLLMProvidersFailed
from app.core.utils import effective_question, log, safe_json
from app.prompts.reviewer_prompt import REVIEWER_PROMPT
from app.tools.citation import select_citable_sources, split_reference_section
from app.tools.source_filter import LOW_QUALITY_DOMAINS
from app.workflows.states import AgentState


# Academic source types — V13 includes openalex and crossref
_ACADEMIC_SOURCE_TYPES = {"arxiv", "semantic_scholar", "alphaxiv", "openalex", "crossref"}

# Score cap when academic_ratio < 0.3
_LOW_ACADEMIC_SCORE_CAP = 0.60


class ReviewRejectedError(RuntimeError):
    """Raised when the draft does not pass the configured review policy."""


def _source_quality_gate(
    sources: List[Dict],
    need_external_search: bool,
    internal_context: List[Dict] | None = None,
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
        if internal_context:
            return True, []
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


def _citation_failures(draft: str, sources: List[Dict]) -> List[str]:
    """Reject citations without a source excerpt or canonical reference entry."""
    if not draft:
        return []
    real_sources = [source for source in sources if source.get("title") != "__research_notes__"]
    citable_indexes = {index for index, _ in select_citable_sources(sources)}
    body, references = split_reference_section(draft)
    body_markers = {int(marker) for marker in re.findall(r"\[(\d+)\]", body)}
    reference_markers = {int(marker) for marker in re.findall(r"(?m)^\s*\[(\d+)\]", references)}
    failures = []
    for marker in sorted(body_markers):
        if marker > len(real_sources):
            failures.append("citation_out_of_range")
        elif marker not in citable_indexes:
            failures.append(f"citation_without_evidence:[{marker}]")
        elif marker not in reference_markers:
            failures.append(f"citation_missing_reference:[{marker}]")
    return failures


_HIGH_SPECIFICITY_IDENTIFIER = re.compile(
    r"\b(?:[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+|"
    r"[a-z]+(?:-[a-z]+)+\s+(?:framework|system|agent|model|method))\b"
)


def _citation_bound_evidence(sources: List[Dict]) -> str:
    """Return bounded excerpts indexed exactly like external citations."""
    excerpts = []
    for index, source in select_citable_sources(sources):
        content = (source.get("content") or "").strip()
        excerpts.append(
            f"[{index}] {source.get('title', 'Untitled')}\n"
            f"Verbatim excerpt: {content[:1200]}"
        )
    return "\n\n".join(excerpts) or "No verbatim external excerpts available."


def _unsupported_cited_identifiers(draft: str, sources: List[Dict]) -> List[str]:
    """Find specific named terms cited to sources whose excerpts do not contain them.

    The check is intentionally limited to CamelCase system names and hyphenated
    named frameworks. Acronyms require semantic context and remain an LLM
    reviewer responsibility. This is a guardrail, not an entailment engine.
    """
    if not draft:
        return []

    citable_sources = dict(select_citable_sources(sources))
    body, _ = split_reference_section(draft)
    failures = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        markers = {int(marker) for marker in re.findall(r"\[(\d+)\]", sentence)}
        if not markers:
            continue
        identifiers = {
            match.group(0)
            for match in _HIGH_SPECIFICITY_IDENTIFIER.finditer(sentence)
        }
        for identifier in identifiers:
            if all(
                marker not in citable_sources
                or identifier.casefold() not in (citable_sources[marker].get("content") or "").casefold()
                for marker in markers
            ):
                failures.append(f"hallucination: unsupported identifier '{identifier}'")
    return list(dict.fromkeys(failures))


_QUANTITATIVE_CLAIM = re.compile(
    r"\b\d+(?:\.\d+)?\s*%|\b\d+\.\d+\b|\b[A-Za-z][A-Za-z0-9_-]*@\d+\b"
)


def _unsupported_quantitative_claims(draft: str, sources: List[Dict]) -> List[str]:
    """Find cited percentages, decimals, and named metrics absent from evidence."""
    if not draft:
        return []
    citable_sources = dict(select_citable_sources(sources))
    body, _ = split_reference_section(draft)
    failures = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        markers = {int(marker) for marker in re.findall(r"\[(\d+)\]", sentence)}
        if not markers:
            continue
        claims = {match.group(0) for match in _QUANTITATIVE_CLAIM.finditer(sentence)}
        for claim in claims:
            normalized_claim = re.sub(r"\s+", "", claim).casefold()
            if all(
                marker not in citable_sources
                or normalized_claim not in re.sub(
                    r"\s+", "", citable_sources[marker].get("content") or ""
                ).casefold()
                for marker in markers
            ):
                failures.append(f"hallucination: unsupported quantitative claim '{claim}'")
    return list(dict.fromkeys(failures))


class ReviewerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        if state.failure_code in {"writer_empty_response", "writer_insufficient_evidence"}:
            state.reviewed_answer = None
            state.review_decision = "rejected"
            state.confidence_score = 0.0
            state.review_feedback = state.failure_message or "Writer model returned an empty response."
            log(state, "[ReviewerAgent] SKIPPED — writer returned an empty response")
            return state

        # V13: Run source quality gate before LLM evaluation
        gate_passed, gate_failed = _source_quality_gate(
            state.external_context,
            state.need_external_search,
            state.vector_context,
        )

        # Compute metrics for logging
        real_sources = [s for s in state.external_context if s.get("title") != "__research_notes__"]
        total_count = len(real_sources)
        academic_count = sum(1 for s in real_sources if s.get("source_type") in _ACADEMIC_SOURCE_TYPES)
        academic_ratio = academic_count / max(total_count, 1)

        log(state, f"\n[ReviewerAgent] Source quality: {academic_count}/{total_count} academic (ratio={academic_ratio:.2f})")
        if gate_failed:
            log(state, f"  [ReviewerAgent] Gate failed: {gate_failed}")
        citation_failures = _citation_failures(
            state.draft_answer or "", state.external_context
        )
        for failure in citation_failures:
            if failure not in gate_failed:
                gate_failed.append(failure)
        unsupported_identifiers = _unsupported_cited_identifiers(
            state.draft_answer or "", state.external_context
        )
        for failure in unsupported_identifiers:
            if failure not in gate_failed:
                gate_failed.append(failure)
        unsupported_quantitative_claims = _unsupported_quantitative_claims(
            state.draft_answer or "", state.external_context
        )
        for failure in unsupported_quantitative_claims:
            if failure not in gate_failed:
                gate_failed.append(failure)

        # Critical: no sources at all when external search was needed
        if "no_sources_found" in gate_failed and state.need_external_search:
            state.confidence_score = 0.0
            state.review_feedback = "No external sources found. Cannot evaluate grounding quality."
            state.iteration_count += 1
            log(state, f"\n[ReviewerAgent] Iteration {state.iteration_count} — REWRITE (no sources)")
            return state

        # Include citation-bound evidence so the reviewer can verify named
        # systems, methods, metrics, and citations against source text.
        source_summary = _build_source_summary(state.external_context)
        source_evidence = _citation_bound_evidence(state.external_context)
        if state.vector_context:
            source_evidence += "\n\n=== Internal PDF evidence ===\n" + "\n".join(
                f"[PDF-{index + 1}] {chunk.get('section', '?')}: {chunk.get('content', '')[:300]}"
                for index, chunk in enumerate(state.vector_context)
            )

        try:
            res = self.llm.invoke([
                SystemMessage(content=REVIEWER_PROMPT),
                HumanMessage(content=(
                    f"Research Question: {effective_question(state)}\n\n"
                    f"=== Sources Used ===\n{source_summary}\n\n"
                    f"=== Citation-Bound Evidence ===\n{source_evidence}\n\n"
                    f"=== Draft ===\n{state.draft_answer}"
                )),
            ])
        except AllLLMProvidersFailed as exc:
            log(state, f"  [WARN] Reviewer unavailable; report is rejected: {exc}")
            state.reviewed_answer = None
            state.confidence_score = 0.0
            state.review_decision = "rejected"
            state.failure_code = "reviewer_unavailable"
            state.review_feedback = "Automated reviewer unavailable; report was not released."
            state.iteration_count += 1
            return state

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
        if citation_failures:
            feedback = (
                f"{feedback}\n\nFix citation mapping: "
                f"{', '.join(citation_failures[:3])}. Use only markers that have both "
                f"citation-bound evidence and a canonical reference entry."
            )
        if unsupported_identifiers:
            unsupported_names = ", ".join(
                failure.split("'")[1] for failure in unsupported_identifiers[:3]
            )
            feedback = (
                f"{feedback}\n\nRemove {unsupported_names} unless the exact identifier "
                f"appears in the citation-bound evidence for its inline citation."
            )
        if unsupported_quantitative_claims:
            unsupported_values = ", ".join(
                failure.split("'")[1] for failure in unsupported_quantitative_claims[:3]
            )
            feedback = (
                f"{feedback}\n\nRemove or correct {unsupported_values}; each quantitative "
                f"value must appear in the evidence for its inline citation."
            )

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
            state.review_decision = "accept"
            log(state, "  -> ACCEPTED")
        else:
            state.review_decision = "rewrite" if state.iteration_count < state.max_iterations else "rejected"
            log(state, f"  -> REWRITE ({state.iteration_count}/{state.max_iterations})")
            if state.iteration_count >= state.max_iterations:
                log(state, "  -> REJECTED — draft remains unreviewed")

        return state
