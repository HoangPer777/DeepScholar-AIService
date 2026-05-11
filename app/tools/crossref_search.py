"""
CrossRef search tool — V13 Academic Search (Fallback Source).

Fallback khi arXiv bị rate-limit (429).
CrossRef là metadata database cho published academic works, free, no auth required.

API docs: https://api.crossref.org/swagger-ui/index.html
"""
from typing import Dict, List, Optional

import requests


_CROSSREF_URL = "https://api.crossref.org/works"
_REQUEST_TIMEOUT = 10  # seconds
# Polite pool: faster rate limit khi có mailto header
_HEADERS = {"User-Agent": "DeepScholar/1.0 (mailto:contact@deepscholar.app)"}


def _extract_year(published: Optional[dict]) -> Optional[int]:
    """Extract year từ CrossRef published field."""
    if not published:
        return None
    try:
        date_parts = published.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            return int(date_parts[0][0])
    except (IndexError, TypeError, ValueError):
        pass
    return None


def _format_author(author: dict) -> str:
    """Format CrossRef author dict thành full name string."""
    given = (author.get("given") or "").strip()
    family = (author.get("family") or "").strip()
    if given and family:
        return f"{given} {family}"
    return family or given


def search_crossref(query: str, max_results: int = 5) -> List[Dict]:
    """
    Tìm kiếm papers qua CrossRef API.

    Preconditions:
    - query là string (có thể rỗng)
    - max_results > 0

    Postconditions:
    - Trả về list (có thể rỗng nếu API fail hoặc không có kết quả)
    - Mỗi entry có source_type = "crossref"
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
            "query": query.strip(),
            "rows":  max_results,
            "select": "title,abstract,URL,author,published,is-referenced-by-count,container-title,DOI",
        }
        resp = requests.get(
            _CROSSREF_URL,
            params=params,
            headers=_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )

        if resp.status_code != 200:
            print(f"[CrossRef] API error: {resp.status_code}")
            return []

        data = resp.json()
        items = data.get("message", {}).get("items", [])
        results = []

        for item in items:
            # Title: CrossRef returns list
            title_list = item.get("title") or []
            title = title_list[0].strip() if title_list else ""
            if not title:
                continue

            # Abstract: may contain JATS XML tags, strip them
            abstract_raw = (item.get("abstract") or "").strip()
            # Simple tag stripping for JATS XML
            import re
            abstract = re.sub(r"<[^>]+>", " ", abstract_raw).strip()
            abstract = re.sub(r"\s+", " ", abstract)[:2000]

            # URL: prefer DOI URL
            doi = item.get("DOI") or ""
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")

            # Authors
            authors = [
                _format_author(a)
                for a in (item.get("author") or [])
                if _format_author(a)
            ]

            # Year
            year = _extract_year(item.get("published"))

            # Venue: container-title is a list
            venue_list = item.get("container-title") or []
            venue = venue_list[0].strip() if venue_list else ""

            results.append({
                "source_type":    "crossref",
                "title":          title,
                "content":        abstract,
                "url":            url,
                "authors":        authors,
                "year":           year,
                "citation_count": item.get("is-referenced-by-count") or 0,
                "venue":          venue,
                "score":          0.90,
            })

        print(f"[CrossRef] {len(results)} results for query: {query[:50]}")
        return results

    except Exception as e:
        print(f"[CrossRef] Error: {e}")
        return []
