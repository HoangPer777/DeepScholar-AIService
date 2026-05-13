"""
ResearcherAgent — V15.

V15 changes:
- Academic minimum guarantee: if < 3 academic sources after initial fetch,
  trigger one re-search with broader query ("survey overview" suffix)
- Query deduplication before fetching
"""
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import effective_question, log
from app.prompts.researcher_prompt import RESEARCHER_PROMPT
from app.tools.academic_search import academic_search
from app.workflows.states import AgentState


_ACADEMIC_SOURCE_TYPES = {"arxiv", "semantic_scholar", "alphaxiv", "openalex", "crossref"}
_MIN_ACADEMIC_SOURCES = 3  # trigger re-search if below this


def _deduplicate_queries(queries: List[str]) -> List[str]:
    seen: set = set()
    result: List[str] = []
    for q in queries:
        normalized = q.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(q)
    return result


def _collect_sources(queries: List[str]) -> List[dict]:
    """Run academic_search for each query, deduplicate by URL."""
    all_results, seen_urls = [], set()
    for q in queries:
        for r in academic_search(q):
            url = r.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
            elif not url:
                all_results.append(r)
    return all_results


class ResearcherAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        if not state.need_external_search:
            log(state, "\n[ResearcherAgent] SKIPPED")
            return state

        # Deduplicate queries
        original_count = len(state.search_queries)
        deduped_queries = _deduplicate_queries(state.search_queries)
        if len(deduped_queries) < original_count:
            log(state, f"[ResearcherAgent] Queries: {original_count} → {len(deduped_queries)} after dedup")

        if not deduped_queries:
            log(state, "[ResearcherAgent] WARNING: no queries after dedup — skipping")
            return state

        # Initial fetch
        all_results = _collect_sources(deduped_queries)
        for q in deduped_queries:
            log(state, f"[ResearcherAgent] Query: '{q}' → collected")

        # Academic minimum guarantee: re-search if < _MIN_ACADEMIC_SOURCES
        academic_count = sum(1 for r in all_results if r.get("source_type") in _ACADEMIC_SOURCE_TYPES)
        if academic_count < _MIN_ACADEMIC_SOURCES and deduped_queries:
            log(state, f"[ResearcherAgent] Only {academic_count} academic sources — triggering re-search")
            # Broaden the first query with "survey overview" to get more academic hits
            base_query = deduped_queries[0]
            broader_query = f"{base_query} survey overview"
            extra = _collect_sources([broader_query])
            # Merge, dedup by URL
            existing_urls = {r.get("url") for r in all_results if r.get("url")}
            for r in extra:
                url = r.get("url") or ""
                if url and url not in existing_urls:
                    existing_urls.add(url)
                    all_results.append(r)
            academic_count = sum(1 for r in all_results if r.get("source_type") in _ACADEMIC_SOURCE_TYPES)
            log(state, f"[ResearcherAgent] After re-search: {len(all_results)} sources ({academic_count} academic)")

        state.external_context = all_results

        # Build numbered input for researcher LLM
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

        state.external_context.insert(0, {
            "title":       "__research_notes__",
            "content":     res.content,
            "url":         "",
            "score":       1.0,
            "source_type": "internal",
        })

        log(state, f"[ResearcherAgent] {len(all_results)} unique sources ({academic_count} academic) — notes extracted")
        return state
