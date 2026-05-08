from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import effective_question, log
from app.prompts.researcher_prompt import RESEARCHER_PROMPT
from app.tools.academic_search import academic_search
from app.workflows.states import AgentState


class ResearcherAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        if not state.need_external_search:
            log(state, "\n[ResearcherAgent] SKIPPED")
            return state

        # V12: Hybrid academic search (Semantic Scholar + arXiv + Tavily)
        all_results, seen = [], set()
        for q in state.search_queries:
            for r in academic_search(q):
                if r["url"] not in seen:
                    seen.add(r["url"])
                    all_results.append(r)
            log(state, f"[ResearcherAgent] Query: '{q}' → collected")

        state.external_context = all_results

        # Build numbered input for researcher LLM
        # Include alphaxiv_url in the source info so agent knows about it
        numbered = "\n\n".join(
            f"[{i + 1}] Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            + (f"AlphaXiv: {r['alphaxiv_url']}\n" if r.get("alphaxiv_url") else "")
            + f"Type: {r.get('source_type', 'web')}\n"
            + (f"Authors: {', '.join(r['authors'][:3])}\n" if r.get("authors") else "")
            + (f"Year: {r['year']}\n" if r.get("year") else "")
            + (f"Citations: {r['citation_count']}\n" if r.get("citation_count") else "")
            + f"Content: {r['content']}"
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

        academic_count = sum(
            1 for r in all_results
            if r.get("source_type") in ("arxiv", "semantic_scholar", "alphaxiv")
        )
        log(state, f"[ResearcherAgent] {len(all_results)} unique sources ({academic_count} academic) — notes extracted")
        return state
