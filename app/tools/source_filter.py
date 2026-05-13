"""
Source filter — V15 Academic Search.

V15 changes:
- Strict domain allowlist: only trusted academic domains per user spec
- Academic source types always bypass domain check (DOI URLs are opaque)
- Web sources (Tavily) always pass as supplementary
"""
from typing import Dict, List


# Social media / low-quality domains — always blocked
LOW_QUALITY_DOMAINS = {
    "reddit.com",
    "quora.com",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
}

# Strict trusted academic domains per user spec
_TRUSTED_DOMAINS = {
    "arxiv.org",
    "doi.org",
    "semanticscholar.org",
    "openalex.org",
    "aclanthology.org",
    "neurips.cc",
    "openreview.net",
    "proceedings.mlr.press",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "alphaxiv.org",
}

# Academic source types — always bypass domain check (DOI URLs may not match list)
_ACADEMIC_SOURCE_TYPES = {"arxiv", "semantic_scholar", "alphaxiv", "openalex", "crossref"}
        if not any(d in (s.get("url") or "").lower() for d in LOW_QUALITY_DOMAINS)
    ]


def filter_by_domain(sources: List[Dict]) -> List[Dict]:
    """
    Keep only sources from trusted academic domains.

    Rules (in priority order):
    1Academic source types (openalex, semantic_scholar, arxiv, crossref) → always pass
    2. Web sources (Tavily, source_type="web") → always pass as supplementary
    3. All others → must match _TRUSTED_DOMAINS
    """
    if not sources:
        return []

    result = []
    for source in sources:
        source_type = source.get("source_type") or "web"
        url = (source.get("url") or "").lower()

def filter_by_domain(sources: List[Dict]) -> List[Dict]:
    """
    Keep only sources from trusted academic domains.

    Rules (in priority order):
    1. Academic source types → always pass (DOI URLs are opaque)
    2. Web sources (source_type="web") → always pass as supplementary
    3. All others → must match _TRUSTED_DOMAINS
    """
    if not sources:
        return []

    result = []
    for source in sources:
        source_type = source.get("source_type") or "web"
        url = (source.get("url") or "").lower()

        if source_type in _ACADEMIC_SOURCE_TYPES:
            result.append(source)
            continue

        if source_type == "web":
            result.append(source)
            continue

        if any(domain in url for domain in _TRUSTED_DOMAINS):
            result.append(source)

    return result


def enforce_source_diversity(sources: List[Dict]) -> List[Dict]:
    """Enforce: academic[:12] + web[:3]."""
    if not sources:
        return []
    academic = [s for s in sources if (s.get("source_type") or "web") in _ACADEMIC_SOURCE_TYPES]
    web = [s for s in sources if (s.get("source_type") or "web") not in _ACADEMIC_SOURCE_TYPES]
    return academic[:12] + web[:3]