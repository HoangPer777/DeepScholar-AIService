from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import effective_question, log, safe_json
from app.prompts.reviewer_prompt import REVIEWER_PROMPT
from app.workflows.states import AgentState


class ReviewerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        res = self.llm.invoke([
            SystemMessage(content=REVIEWER_PROMPT),
            HumanMessage(content=(
                f"Research Question: {effective_question(state)}\n\n"
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
        failed = data.get("failed_criteria", [])
        feedback = data.get("feedback", "No feedback.")

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
            # Force accept at max iterations to always produce output
            if state.iteration_count >= state.max_iterations:
                state.reviewed_answer = state.draft_answer
                log(state, "  -> MAX ITERATIONS reached — using best draft")

        return state
