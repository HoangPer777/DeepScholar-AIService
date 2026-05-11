"""
OpenAlex search tool — V13 Academic Search (Fallback Source).

Fallback khi Semantic Scholar bị rate-limit (429).
OpenAlex là open-access academic database, free, no auth required.
Rate limit: 10 req/s (polite pool với email header).

API docs: https://docs.openalex.org/api-entities/works/search-works
"""
from typing import Dict, List

import requests


_OPENALEX_URL = "https://api.openalex.org/works"
_REQUEST_TIMEOUT = 10  # seconds
# Polite pool: faster rate limit khi có email header
_HEADERS = {"User-Agent": "DeepScholar/1.0 (mailto:contact@deepscholar.app)"}


def _reconstruct_abstract(inverted_index: dict) -> str:
    """
    Reconstruct abstract từ OpenAlex abstract_inverted_index format.

    OpenAlex lưu abstract dưới dạng inverted index:
    {"word": [position1, position2, ...], ...}

    Preconditions:
    - inverted_index là dict (có thể rỗng hoặc None)

    Postconditions:
    - Trả về string (có thể rỗng)
    - Không raise exception
    """
    if not inverted_index:
        return ""
    try:
        # Build position → word mapping
        pos_word: Dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                pos_word[pos] = word
        if not pos_word:
            return ""
        # Sort by position and join
        words = [pos_word[i] for i in sorted(pos_word.keys())]
        return " ".join(words)[:2000]
    except Exception:
        return ""


def search_openalex(query: str, max_results: int = 5) -> List[Dict]:
    """
    Tìm kiếm papers qua OpenAlex API.

    Preconditions:
    - query là string (có thể rỗng)
    - max_results > 0

    Postconditions:
    - Trả về list (có thể rỗng nếu API fail hoặc không có kết quả)
    - Mỗi entry có source_type = "openalex"
    - Không raise exception (caller không cần try/except)

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
            "search":   query.strip(),
            "per-page": max_results,
            "select":   "id,display_name,abstract_inverted_index,authorships,publication_year,cited_by_count,primary_location,doi",
        }
        resp = requests.get(
            _OPENALEX_URL,
            params=params,
            headers=_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            print(f"[OpenAlex] API error: {resp.status_code}")
            return []

        data = resp.json()
        results = []

        for work in data.get("results", []):
            title = (work.get("display_name") or "").strip()
            if not title:
                continue

            # Reconstruct abstract from inverted index
            abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))

            # URL: prefer DOI, fallback to OpenAlex ID
            doi = work.get("doi") or ""
            openalex_id = work.get("id") or ""
            url = doi if doi else openalex_id

            # Authors from authorships
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in (work.get("authorships") or [])
                if a.get("author", {}).get("display_name")
            ]

            # Venue from primary_location
            venue = ""
            primary_loc = work.get("primary_location") or {}
            source = primary_loc.get("source") or {}
            venue = (source.get("display_name") or "").strip()

            results.append({
                "source_type":    "openalex",
                "title":          title,
                "content":        abstract[:2000] if abstract else "",
                "url":            url,
                "authors":        authors,
                "year":           work.get("publication_year"),
                "citation_count": work.get("cited_by_count") or 0,
                "venue":          venue,
                "score":          0.95,
            })

        print(f"[OpenAlex] {len(results)} results for query: {query[:50]}")
        return results

    except Exception as e:
        print(f"[OpenAlex] Error: {e}")
        return []
