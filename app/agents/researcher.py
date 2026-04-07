from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import effective_question, log
from app.prompts.researcher_prompt import RESEARCHER_PROMPT
from app.tools.citation import enrich_arxiv_metadata
from app.tools.tavily_search import tavily_search
from app.workflows.states import AgentState


class ResearcherAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        if not state.need_external_search:
            log(state, "\n[ResearcherAgent] SKIPPED")
            return state

        all_results, seen = [], set()
        for q in state.search_queries:
            for r in tavily_search(q, max_results=5):
                if r["url"] not in seen:
                    seen.add(r["url"])
                    all_results.append(r)
            log(state, f"[ResearcherAgent] Query: '{q}' → collected")

        log(state, "[ResearcherAgent] Enriching arxiv sources via Semantic Scholar...")
        all_results = enrich_arxiv_metadata(all_results)

        state.external_context = all_results

        # Build numbered input for researcher LLM
        numbered = "\n\n".join(
            f"[{i + 1}] Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for i, r in enumerate(all_results)
        )

        res = self.llm.invoke([
            SystemMessage(content=RESEARCHER_PROMPT),
            HumanMessage(content=f"Research question: {effective_question(state)}\n\nSources:\n{numbered}"),
        ])

        # Insert research notes at the front of external_context
        state.external_context.insert(0, {
            "title":       "__research_notes__",
            "content":     res.content,
            "url":         "",
            "score":       1.0,
            "source_type": "internal",
        })

        arxiv_count = sum(1 for r in all_results if r.get("source_type") == "arxiv")
        log(state, f"[ResearcherAgent] {len(all_results)} unique sources ({arxiv_count} arxiv enriched) — notes extracted")
        return state
