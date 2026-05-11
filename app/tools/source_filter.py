"""
Source filter — V14 Academic Search.

V14 changes:
- filter_by_domain(): trusted academic domain allowlist
- enforce_source_diversity: includes openalex/crossref as academic
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

# Trusted academic domains — sources from these pass domain filter
# Web sources (Tavily) bypass this check (source_type == "web")
_TRUSTED_ACADEMIC_DOMAINS = {
    "arxiv.org",
    "semanticscholar.org",
    "doi.org",
    "openalex.org",
    "crossref.org",
    "neurips.cc",
    "aclanthology.org",
    "openreview.net",
    "proceedings.mlr.press",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "springer.com",
    "nature.com",
    "science.org",
    "plos.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "alphaxiv.org",
    "huggingface.co",  # model cards / papers
    "github.com",      # code repos
}

# Academic source types — these bypass domain check (URL may be opaque DOI)
_ACADEMIC_SOURCE_TYPES = {"arxiv", "semantic_scholar", "alphaxiv", "openalex", "crossref"}


def filter_low_quality_sources(sources: List[Dict]) -> List[Dict]:
    """Remove sources from social media / low-quality domains."""
    if not sources:
        return []
    return [
        s for s in sources
        if not any(d in (s.get("url") or "").lower() for d in LOW_QUALITY_DOMAINS)
    ]


def filter_by_domain(sources: List[Dict]) -> List[Dict]:
    """
    Keep only sources from trusted academic domains.

    Rules:
    - Academic source types (openalex, semantic_scholar, arxiv, crossref) always pass
      regardless of URL (DOI URLs may not match domain list)
    - Web sources (Tavily) always pass — they are supplementary
    - Other sources must have URL matching a trusted domain

    Postconditions:
    - Sources with source_type in _ACADEMIC_SOURCE_TYPES always included
    - Sources with source_type == "web" always included (Tavily supplement)
    - Other sources only included if URL matches _TRUSTED_ACADEMIC_DOMAINS
    - Không raise exception
    """
    if not sources:
        return []

    result = []
    for source in sources:
        source_type = source.get("source_type") or "web"
        url = (source.get("url") or "").lower()

        # Academic sources always pass (DOI URLs may not match domain list)
        if source_type in _ACADEMIC_SOURCE_TYPES:
            result.append(source)
            continue

        # Web sources (Tavily) always pass as supplementary
        if source_type == "web":
            result.append(source)
            continue

        # Other sources: check domain allowlist
        if any(domain in url for domain in _TRUSTED_ACADEMIC_DOMAINS):
            result.append(source)

    return result


def enforce_source_diversity(sources: List[Dict]) -> List[Dict]:
    """
    Enforce source diversity: academic[:12] + web[:3].

    Invariant:
    - len(academic) <= 12
    - len(web) <= 3
    """
    if not sources:
        return []

    academic = [s for s in sources if (s.get("source_type") or "web") in _ACADEMIC_SOURCE_TYPES]
    web = [s for s in sources if (s.get("source_type") or "web") not in _ACADEMIC_SOURCE_TYPES]

    return academic[:12] + web[:3]
