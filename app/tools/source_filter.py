"""
Source filter — V12 Academic Search.

Lọc low-quality sources và enforce diversity:
- filter_low_quality_sources: Loại bỏ reddit, quora, linkedin, facebook, twitter, x.com
- enforce_source_diversity: Ưu tiên academic sources (12) vs web sources (3)
"""
from typing import Dict, List


# Low-quality domains to filter out
LOW_QUALITY_DOMAINS = {
    "reddit.com",
    "quora.com",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
}


def filter_low_quality_sources(sources: List[Dict]) -> List[Dict]:
    """
    Loại bỏ sources từ low-quality domains.

    Preconditions:
    - sources là list of dicts (có thể rỗng)

    Postconditions:
    - Trả về filtered list (có thể rỗng)
    - Sources từ LOW_QUALITY_DOMAINS bị loại bỏ
    - Không raise exception

    Args:
        sources: List of source dicts

    Returns:
        Filtered list without low-quality sources
    """
    if not sources:
        return []

    filtered = []
    for source in sources:
        url = (source.get("url") or "").lower()

        # Check if URL contains any low-quality domain
        if any(bad_domain in url for bad_domain in LOW_QUALITY_DOMAINS):
            continue

        filtered.append(source)

    return filtered


def enforce_source_diversity(sources: List[Dict]) -> List[Dict]:
    """
    Enforce source diversity: academic[:12] + web[:3].

    Preconditions:
    - sources là list of dicts (có thể rỗng)

    Postconditions:
    - Trả về list với len(academic) <= 12, len(web) <= 3
    - academic = arxiv + semantic_scholar + alphaxiv
    - web = tất cả source types khác
    - Không raise exception

    Invariant:
    - len([s for s in result if s["source_type"] in ("arxiv", "semantic_scholar", "alphaxiv")]) <= 12
    - len([s for s in result if s["source_type"] not in ("arxiv", "semantic_scholar", "alphaxiv")]) <= 3

    Args:
        sources: List of source dicts

    Returns:
        List with enforced diversity (academic[:12] + web[:3])
    """
    if not sources:
        return []

    academic = []
    web = []

    for source in sources:
        source_type = source.get("source_type") or "web"

        if source_type in ("arxiv", "semantic_scholar", "alphaxiv"):
            academic.append(source)
        else:
            web.append(source)

    # Enforce limits
    return academic[:12] + web[:3]
