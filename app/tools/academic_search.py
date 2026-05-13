"""
Hybrid Academic Search — V14.

Pipeline: Semantic Scholar + OpenAlex (primary) + CrossRef + Tavily
- arXiv direct scraping REMOVED (unstable, frequent timeouts)
- OpenAlex replaces arXiv: indexes arXiv, CrossRef, DOI, authors, citations
- Domain filtering: only trusted academic domains pass through
- Relevance scoring added to reranker

V14 changes vs V13:
- arXiv direct fetch removed → OpenAlex promoted to primary source
- New pipeline: SS (primary) + OpenAlex (primary) + CrossRef (supplementary) + Tavily
- Domain allowlist filtering added after dedup
- Relevance scoring in reranker (title/abstract match weight)
"""
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Callable, Dict, List, Optional, Tuple

from app.tools.crossref_search import search_crossref
from app.tools.openalex_search import search_openalex
from app.tools.semantic_scholar import RateLimitError, search_semantic_scholar
from app.tools.source_filter import enforce_source_diversity, filter_low_quality_sources, filter_by_domain
from app.tools.source_reranker import rerank_sources
from app.tools.tavily_search import tavily_search


# Parallel fetch timeout
_PARALLEL_TIMEOUT = 20.0
# Retry: fail-fast to fallback quickly
_MAX_RETRIES = 2
_BASE_DELAY = 0.5  # 0.5s, 1s


def _fetch_with_retry(
    fetch_fn: Callable,
    fallback_fn: Optional[Callable],
    query: str,
    source_name: str,
    max_retries: int = _MAX_RETRIES,
    base_delay: float = _BASE_DELAY,
) -> Tuple[str, List[Dict]]:
    """Retry fetch_fn on RateLimitError, then call fallback_fn if exhausted."""
    for attempt in range(max_retries):
        try:
            results = fetch_fn(query)
            print(f"[AcademicSearch] {source_name}: {len(results)} results (attempt {attempt + 1})")
            return (source_name, results)
        except RateLimitError:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[AcademicSearch] {source_name}: 429, retry in {delay:.1f}s ({attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"[AcademicSearch] {source_name}: exhausted retries — trying fallback")
        except Exception as e:
            print(f"[AcademicSearch] {source_name}: error: {e}")
            break

    if fallback_fn is not None:
        try:
            results = fallback_fn(query)
            print(f"[AcademicSearch] {source_name} fallback: {len(results)} results")
            return (source_name, results)
        except Exception as e:
            print(f"[AcademicSearch] {source_name} fallback error: {e}")

    return (source_name, [])


def _parallel_fetch(query: str, timeout: float = _PARALLEL_TIMEOUT) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Fetch từ tất cả sources đồng thời.

    V14 sources:
    - SemanticScholar (primary, fallback: OpenAlex)
    - OpenAlex (primary — indexes arXiv/CrossRef/DOI, replaces direct arXiv scraping)
    - CrossRef (supplementary)
    - Tavily (web supplement)
    """
    if not query or not query.strip():
        return [], {}

    tasks = [
        (
            lambda q: search_semantic_scholar(q, max_results=5),
            lambda q: search_openalex(q, max_results=5),
            "SemanticScholar",
        ),
        (
            lambda q: search_openalex(q, max_results=5),
            lambda q: search_crossref(q, max_results=5),
            "OpenAlex",
        ),
        (
            lambda q: tavily_search(q, max_results=3),
            None,
            "Tavily",
        ),
    ]

    all_results: List[Dict] = []
    source_counts: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(_fetch_with_retry, fetch_fn, fallback_fn, query, name): name
            for fetch_fn, fallback_fn, name in tasks
        }

        done, not_done = wait(future_to_name.keys(), timeout=timeout)

        for future in not_done:
            future.cancel()
            name = future_to_name[future]
            print(f"[AcademicSearch] {name}: timed out after {timeout:.0f}s — skipped")

        for future in done:
            try:
                source_name, results = future.result()
                if source_name == "Tavily":
                    for r in results:
                        r.setdefault("source_type", "web")
                        r.setdefault("citation_count", 0)
                source_counts[source_name] = len(results)
                all_results.extend(results)
            except Exception as e:
                name = future_to_name[future]
                print(f"[AcademicSearch] {name}: error collecting result: {e}")

    return all_results, source_counts


def academic_search(query: str) -> List[Dict]:
    """
    Hybrid academic search pipeline — V14.

    Pipeline:
    1. Parallel fetch: SS + OpenAlex + Tavily (with fallbacks)
    2. filter_low_quality_sources() — remove social media
    3. filter_by_domain() — keep only trusted academic domains
    4. Deduplicate by URL
    5. rerank_sources() — keyword + citation + source_type + relevance
    6. enforce_source_diversity()
    """
    all_results, source_counts = _parallel_fetch(query)

    if not all_results:
        print(f"[AcademicSearch] No results for: {query[:50]}")
        return []

    # Filter social media / low-quality
    filtered = filter_low_quality_sources(all_results)

    # Domain filtering — keep trusted academic domains + allow web for Tavily
    filtered = filter_by_domain(filtered)

    # Deduplicate by URL
    deduped: List[Dict] = []
    seen_urls: set = set()
    for source in filtered:
        url = source.get("url") or ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(source)
        elif not url:
            deduped.append(source)

    # Rerank
    ranked = rerank_sources(deduped, query)

    # Enforce diversity
    final = enforce_source_diversity(ranked)

    academic_types = {"arxiv", "semantic_scholar", "alphaxiv", "openalex", "crossref"}
    academic_count = sum(1 for s in final if s.get("source_type") in academic_types)
    web_count = len(final) - academic_count

    print(
        f"[AcademicSearch] Final: {len(final)} sources "
        f"({academic_count} academic, {web_count} web) | "
        f"SS={source_counts.get('SemanticScholar', 0)} "
        f"OA={source_counts.get('OpenAlex', 0)} "
        f"Tavily={source_counts.get('Tavily', 0)}"
    )

    return final
