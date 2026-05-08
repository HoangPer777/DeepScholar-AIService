"""
Source reranker — V12 Academic Search.

Rerank sources theo composite score:
- Keyword match: số từ query xuất hiện trong title + content
- Citation bonus: log(1 + citation_count) * 0.1
- Source type bonus: semantic_scholar=0.15, arxiv=0.20, web=0.0
"""
import math
from typing import Dict, List


# Source type bonus weights
_SOURCE_TYPE_BONUS: Dict[str, float] = {
    "semantic_scholar": 0.15,
    "arxiv":            0.20,
    "alphaxiv":         0.18,
    "github":           0.05,
    "blog":             -0.05,
    "web":              0.0,
}


def rerank_sources(sources: List[Dict], query: str) -> List[Dict]:
    """
    Rerank sources theo composite score.

    Preconditions:
    - sources là list of dicts (có thể rỗng)
    - query là string (có thể rỗng)

    Postconditions:
    - Trả về cùng sources được sort theo composite_score descending
    - Mỗi source có field "score" được cập nhật
    - Không raise exception

    Scoring formula:
    score = keyword_score + citation_bonus + source_type_bonus

    Args:
        sources: List of source dicts
        query: Search query string for keyword matching

    Returns:
        Sorted list of sources (highest score first)
    """
    if not sources:
        return []

    query_words = set(query.lower().split()) if query else set()

    for source in sources:
        # Keyword match score
        if query_words:
            text = (
                (source.get("title") or "") + " " +
                (source.get("content") or "")
            ).lower()
            words_in_text = sum(1 for w in query_words if w in text)
            keyword_score = words_in_text / len(query_words)
        else:
            keyword_score = 0.0

        # Citation bonus: log(1 + citation_count) * 0.1
        citation_count = source.get("citation_count") or 0
        citation_bonus = math.log1p(citation_count) * 0.1

        # Source type bonus
        source_type = source.get("source_type") or "web"
        type_bonus = _SOURCE_TYPE_BONUS.get(source_type, 0.0)

        # Composite score
        source["score"] = round(keyword_score + citation_bonus + type_bonus, 4)

    return sorted(sources, key=lambda x: x.get("score", 0.0), reverse=True)
