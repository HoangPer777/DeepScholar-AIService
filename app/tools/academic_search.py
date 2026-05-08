"""
Hybrid Academic Search — V12.

Pipeline kết hợp Semantic Scholar + arXiv + Tavily:
1. Fetch từ Semantic Scholar (5 results)
2. Fetch từ arXiv (5 results, full abstract)
3. Fetch từ Tavily (3 results, web/blog/GitHub)
4. Filter low-quality sources
5. Rerank by composite score
6. Enforce source diversity (academic[:12] + web[:3])
"""
from typing import Dict, List

from app.tools.arxiv_search import search_arxiv
from app.tools.semantic_scholar import search_semantic_scholar
from app.tools.source_filter import enforce_source_diversity, filter_low_quality_sources
from app.tools.source_reranker import rerank_sources
from app.tools.tavily_search import tavily_search


def academic_search(query: str) -> List[Dict]:
    """
    Hybrid academic search pipeline.

    Preconditions:
    - query là string (có thể rỗng)

    Postconditions:
    - Trả về list (có thể rỗng, không raise exception)
    - academic sources <= 12, web sources <= 3
    - Sources được sort theo composite score descending

    Pipeline:
    1. Semantic Scholar search (5 results)
    2. arXiv search (5 results, full abstract)
    3. Tavily web search (3 results)
    4. filter_low_quality_sources()
    5. Deduplicate by URL
    6. rerank_sources()
    7. enforce_source_diversity()

    Args:
        query: Search query string

    Returns:
        List of ranked, diverse source dicts
    """
    # Step 1: Semantic Scholar
    try:
        ss_results = search_semantic_scholar(query, max_results=5)
        print(f"[AcademicSearch] SemanticScholar: {len(ss_results)} results")
    except Exception as e:
        print(f"[AcademicSearch] SemanticScholar error: {e}")
        ss_results = []

    # Step 2: arXiv (full abstract via feedparser)
    try:
        arxiv_results = search_arxiv(query, max_results=5)
        print(f"[AcademicSearch] arXiv: {len(arxiv_results)} results")
    except Exception as e:
        print(f"[AcademicSearch] arXiv error: {e}")
        arxiv_results = []

    # Step 3: Tavily web search (supplementary)
    try:
        web_results = tavily_search(query, max_results=3)
        # Ensure web source_type is set
        for r in web_results:
            r.setdefault("source_type", "web")
            r.setdefault("citation_count", 0)
        print(f"[AcademicSearch] Tavily: {len(web_results)} results")
    except Exception as e:
        print(f"[AcademicSearch] Tavily error: {e}")
        web_results = []

    # Step 4: Merge all results
    all_results = ss_results + arxiv_results + web_results

    if not all_results:
        return []

    # Step 5: Filter low-quality sources
    filtered = filter_low_quality_sources(all_results)

    # Step 6: Deduplicate by URL
    deduped: List[Dict] = []
    seen_urls: set = set()
    for source in filtered:
        url = source.get("url") or ""
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(source)
        elif not url:
            # Include sources without URL (shouldn't happen but be safe)
            deduped.append(source)

    # Step 7: Rerank by composite score
    ranked = rerank_sources(deduped, query)

    # Step 8: Enforce source diversity
    final = enforce_source_diversity(ranked)

    # Log summary
    academic_count = sum(
        1 for s in final
        if s.get("source_type") in ("arxiv", "semantic_scholar", "alphaxiv")
    )
    web_count = len(final) - academic_count
    print(f"[AcademicSearch] Final: {len(final)} sources ({academic_count} academic, {web_count} web)")

    return final
