from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import log, safe_json
from app.prompts.clarifier_prompt import CLARIFIER_PROMPT
from app.workflows.states import AgentState


class ClarifierAgent: 
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        if not state.need_clarification:
            log(state, "\n[ClarifierAgent] SKIPPED — question is clear")
            return state

        res = self.llm.invoke([
            SystemMessage(content=CLARIFIER_PROMPT),
            HumanMessage(content=state.question),
        ])
        data = safe_json(res.content)

        state.clarified_question = data.get("clarified_question", state.question)
        log(state, "\n[ClarifierAgent]")
        log(state, f"  interpretation     : {data.get('interpretation', '')}")
        log(state, f"  clarified_question : {state.clarified_question}")
        return state
