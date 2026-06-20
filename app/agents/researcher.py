"""
ResearcherAgent — V16.

V16 changes:
- Parallel search: replaced sequential _collect_sources loop with
  _collect_sources_parallel using ThreadPoolExecutor (Requirement 1.2, 1.3)
- Query cap: MAX_QUERIES = 5 enforced before executor creation (Requirement 1.4)
- Thread-safe URL deduplication via threading.Lock
- Per-future exception handling with WARNING log; continues on partial failure

V15 changes:
- Academic minimum guarantee: if < 3 academic sources after initial fetch,
  trigger one re-search with broader query ("survey overview" suffix)
- Query deduplication before fetching
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.utils import effective_question, log
from app.prompts.researcher_prompt import RESEARCHER_PROMPT
from app.tools.academic_search import academic_search
from app.workflows.states import AgentState

logger = logging.getLogger(__name__)

MAX_QUERIES = 5  # Requirement 1.4: cap at 5 queries


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
    """Run academic_search for each query, deduplicate by URL (sequential)."""
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


def _collect_sources_parallel(queries: List[str]) -> List[dict]:
    """
    Run academic_search for each query in parallel using ThreadPoolExecutor.
    Deduplicates results by URL. Caps at MAX_QUERIES queries.

    Uses threading.Lock for thread-safe URL deduplication.
    Catches per-future exceptions and logs WARNING, continuing with successful results.
    Emits DEBUG log when query list is truncated to cap.
    """
    if len(queries) > MAX_QUERIES:
        logger.debug(
            "Query list truncated from %d to %d (MAX_QUERIES cap)",
            len(queries),
            MAX_QUERIES,
        )
    capped = queries[:MAX_QUERIES]

    all_results: List[dict] = []
    seen_urls: set = set()
    lock = threading.Lock()

    def fetch_one(q: str) -> List[dict]:
        return academic_search(q)

    with ThreadPoolExecutor(max_workers=min(len(capped), MAX_QUERIES)) as executor:
        futures = {executor.submit(fetch_one, q): q for q in capped}
        for future in as_completed(futures):
            query = futures[future]
            try:
                results = future.result()
                with lock:
                    for r in results:
                        url = r.get("url") or ""
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(r)
                        elif not url:
                            all_results.append(r)
            except Exception as exc:
                logger.warning(
                    "academic_search failed for query '%s': %s", query, exc
                )
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

        # Initial fetch (parallel)
        t0 = time.perf_counter()
        all_results = _collect_sources_parallel(deduped_queries)
        state.timings["external_search_collect_ms"] = int((time.perf_counter() - t0) * 1000)
        for q in deduped_queries:
            log(state, f"[ResearcherAgent] Query: '{q}' → collected")

        # Academic minimum guarantee: re-search if < _MIN_ACADEMIC_SOURCES
        academic_count = sum(1 for r in all_results if r.get("source_type") in _ACADEMIC_SOURCE_TYPES)
        if academic_count < _MIN_ACADEMIC_SOURCES and deduped_queries:
            log(state, f"[ResearcherAgent] Only {academic_count} academic sources — triggering re-search")
            # Broaden the first query with "survey overview" to get more academic hits
            base_query = deduped_queries[0]
            broader_query = f"{base_query} survey overview"
            t0 = time.perf_counter()
            extra = _collect_sources([broader_query])
            state.timings["external_search_retry_ms"] = int((time.perf_counter() - t0) * 1000)
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

        t0 = time.perf_counter()
        res = self.llm.invoke([
            SystemMessage(content=RESEARCHER_PROMPT),
            HumanMessage(content=f"Research question: {effective_question(state)}\n\nSources:\n{numbered}"),
        ])
        state.timings["researcher_llm_ms"] = int((time.perf_counter() - t0) * 1000)

        state.external_context.insert(0, {
            "title":       "__research_notes__",
            "content":     res.content,
            "url":         "",
            "score":       1.0,
            "source_type": "internal",
        })

        log(state, f"[ResearcherAgent] {len(all_results)} unique sources ({academic_count} academic) — notes extracted")
        return state
