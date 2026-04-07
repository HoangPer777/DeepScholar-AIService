from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import log, safe_json
from app.prompts.planner_prompt import PLANNER_PROMPT
from app.workflows.states import AgentState


class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        response = self.llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=state.question),
        ])
        data = safe_json(response.content)

        state.need_clarification = data.get("need_clarification", False)
        state.need_external_search = data.get("need_external_search", True)
        state.focus_sections = data.get("focus_sections", [])
        state.search_queries = data.get("search_queries", [state.question])

        log(state, f"\n{'=' * 60}\n[PlannerAgent]")
        log(state, f"  need_clarification   : {state.need_clarification}")
        log(state, f"  need_external_search : {state.need_external_search}")
        log(state, f"  focus_sections       : {state.focus_sections}")
        log(state, f"  search_queries       : {state.search_queries}")
        return state
