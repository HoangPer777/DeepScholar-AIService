"""
Source reranker — V14 Academic Search.

V14 changes:
- Relevance scoring: title match weighted 2x vs content match
- Phrase match bonus: consecutive query words in title
- Source type bonus: strict academic priority
- blog/medium/towardsdatascience penalty
- Minimum relevance threshold: drop off-topic sources (topical filtering)
"""
import math
from typing import Dict, List


# Source type bonus weights
_SOURCE_TYPE_BONUS: Dict[str, float] = {
    "arxiv":            0.40,
    "semantic_scholar": 0.35,
    "openalex":         0.35,
    "crossref":         0.30,
    "alphaxiv":         0.38,
    "github":           0.05,
    "blog":            -0.10,
    "web":              0.0,
}

# Web domains that get a penalty (low-quality but not blocked)
_WEB_PENALTY_DOMAINS = {
    "medium.com",
    "towardsdatascience.com",
    "analyticsvidhya.com",
    "kdnuggets.com",
    "machinelearningmastery.com",
    "towards-ai.net",
}

# Sources with relevance_score below this threshold are dropped as off-topic.
# 0.05 is permissive enough to keep technical papers with short titles
# (e.g. "Dense Passage Retrieval" matches 1/4 query words → score ≈ 0.08 → kept)
# but strict enough to drop truly unrelated papers (0 word match → score = 0.0 → dropped)
_MIN_RELEVANCE_THRESHOLD = 0.05


def _relevance_score(source: Dict, query: str, query_words: set) -> float:
    """
    Compute relevance score based on query match in title and content.

    Scoring:
    - Title word match: weighted 2x (title is more signal-dense)
    - Content word match: weighted 1x
    - Phrase match bonus: +0.15 if 2+ consecutive query words appear in title
    - Normalized to [0, 1] range
    """
    if not query_words:
        return 1.0  # no query → don't filter anything

    title = (source.get("title") or "").lower()
    content = (source.get("content") or "").lower()

    title_hits = sum(1 for w in query_words if w in title)
    content_hits = sum(1 for w in query_words if w in content)

    n = len(query_words)
    raw = (title_hits * 2 + content_hits) / (n * 3)

    # Phrase match bonus
    words = query.lower().split()
    phrase_bonus = 0.0
    for i in range(len(words) - 1):
        bigram = words[i] + " " + words[i + 1]
        if bigram in title:
            phrase_bonus = 0.15
            break

    return min(raw + phrase_bonus, 1.0)


def _domain_penalty(source: Dict) -> float:
    """Return penalty for low-quality web domains."""
    url = (source.get("url") or "").lower()
    if any(d in url for d in _WEB_PENALTY_DOMAINS):
        return -0.15
    return 0.0


def rerank_sources(
    sources: List[Dict],
    query: str,
    min_relevance: float = _MIN_RELEVANCE_THRESHOLD,
) -> List[Dict]:
    """
    Rerank sources theo composite score, dropping off-topic sources.

    Scoring formula:
    score = relevance_score + citation_bonus + source_type_bonus + domain_penalty

    Topical filtering:
    Sources with relevance_score < min_relevance are dropped before sorting.
    This removes semantic drift (e.g. "nephrology" paper when query is "RAG").

    Args:
        sources: List of source dicts
        query: Search query string
        min_relevance: Minimum relevance score to keep a source (default 0.15)

    Returns:
        Sorted, topically-filtered list of sources (highest score first)
    """
    if not sources:
        return []

    query_words = set(query.lower().split()) if query else set()
    kept = []
    dropped = 0

    for source in sources:
        relevance = _relevance_score(source, query, query_words)

        # Drop off-topic sources before scoring
        if relevance < min_relevance:
            dropped += 1
            continue

        citation_count = source.get("citation_count") or 0
        citation_bonus = math.log1p(citation_count) * 0.1

        source_type = source.get("source_type") or "web"
        type_bonus = _SOURCE_TYPE_BONUS.get(source_type, 0.0)

        domain_pen = _domain_penalty(source)

        source["score"] = round(relevance + citation_bonus + type_bonus + domain_pen, 4)
        kept.append(source)

    if dropped > 0:
        print(f"[Reranker] Dropped {dropped} off-topic sources (relevance < {min_relevance})")

    return sorted(kept, key=lambda x: x.get("score", 0.0), reverse=True)
