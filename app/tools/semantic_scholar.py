"""
Semantic Scholar search tool — V13 Academic Search.

Tìm kiếm papers qua Semantic Scholar Graph API.
Trả về title, abstract, authors, year, venue, citation_count, url.

V13 changes:
- Raise RateLimitError on HTTP 429 (caller handles retry/backoff)
"""
from typing import Dict, List

import requests


_SS_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS = "title,abstract,authors,year,citationCount,url,venue"
_REQUEST_TIMEOUT = 10  # seconds


class RateLimitError(Exception):
    """Raised when Semantic Scholar API returns HTTP 429 Too Many Requests."""
    pass


def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict]:
    """
    Tìm kiếm papers qua Semantic Scholar Graph API.

    Preconditions:
    - query là string (có thể rỗng)
    - max_results > 0

    Postconditions:
    - Trả về list (có thể rỗng nếu API fail hoặc không có kết quả)
    - Mỗi entry có source_type = "semantic_scholar"
    - Mỗi entry có content = abstract[:1500] (đủ để agent đọc và trích dẫn)
    - Không raise exception

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        List of source dicts with academic metadata
    """
    if not query or not query.strip():
        return []

    try:
        params = {
            "query": query.strip(),
            "limit": max_results,
            "fields": _SS_FIELDS,
        }
        resp = requests.get(_SS_API_URL, params=params, timeout=_REQUEST_TIMEOUT)

        if resp.status_code == 429:
            raise RateLimitError(f"Semantic Scholar rate limit (429)")
        if resp.status_code != 200:
            print(f"[SemanticScholar] API error: {resp.status_code}")
            return []

        data = resp.json()
        results = []

        for paper in data.get("data", []):
            abstract = (paper.get("abstract") or "").strip()
            url = paper.get("url") or ""
            title = (paper.get("title") or "").strip()

            # Skip papers without meaningful content
            if not title:
                continue

            authors = [
                a.get("name", "")
                for a in paper.get("authors", [])
                if a.get("name")
            ]

            results.append({
                "source_type":    "semantic_scholar",
                "title":          title,
                "content":        abstract[:1500] if abstract else "",
                "url":            url,
                "authors":        authors,
                "year":           paper.get("year"),
                "citation_count": paper.get("citationCount") or 0,
                "venue":          paper.get("venue") or "",
                "score":          1.0,
            })

        return results

    except RateLimitError:
        raise  # propagate to caller for retry handling
    except Exception as e:
        print(f"[SemanticScholar] Error: {e}")
        return []
