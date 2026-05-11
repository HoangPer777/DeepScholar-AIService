"""
arXiv search tool — V13 Academic Search.

Tìm kiếm preprints qua arXiv API với feedparser.
Trả về abstract đầy đủ (~2000 chars) để agent có thể đọc và trích dẫn.

V13 changes:
- Raise RateLimitError on HTTP 429 (caller handles retry/backoff)
"""
from typing import Dict, List

import feedparser
import requests

from app.tools.semantic_scholar import RateLimitError  # shared exception


_ARXIV_API_URL = "http://export.arxiv.org/api/query"
_REQUEST_TIMEOUT = 8  # seconds — reduced from 10 to fail-fast when rate limited


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Tìm kiếm preprints qua arXiv API với feedparser.

    Preconditions:
    - query là string (có thể rỗng)
    - max_results > 0

    Postconditions:
    - Trả về list (có thể rỗng nếu API fail hoặc không có kết quả)
    - Mỗi entry có source_type = "arxiv"
    - Mỗi entry có content = entry.summary[:2000] (abstract đầy đủ)
    - Mỗi entry có alphaxiv_url = https://www.alphaxiv.org/abs/{arxiv_id}
    - Không raise exception

    QUAN TRỌNG:
    entry.summary từ feedparser là abstract đầy đủ (~2000 chars).
    Đây là "content" chính để agent đọc và trích dẫn.
    Không được cắt ngắn dưới 1500 chars nếu summary đủ dài.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        List of source dicts with arXiv metadata and full abstracts
    """
    if not query or not query.strip():
        return []

    try:
        # Build arXiv API URL
        url = (
            f"{_ARXIV_API_URL}?"
            f"search_query=all:{query.strip()}"
            f"&start=0"
            f"&max_results={max_results}"
        )

        response = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if response.status_code == 429:
            raise RateLimitError(f"arXiv rate limit (429)")
        if response.status_code != 200:
            print(f"[arXiv] API error: {response.status_code}")
            return []

        # Parse Atom feed
        feed = feedparser.parse(response.text)
        results = []
        seen_ids = set()

        for entry in feed.entries:
            # Extract arXiv ID from URL
            arxiv_url = entry.link
            arxiv_id = arxiv_url.split("/")[-1]

            # Deduplicate by arXiv ID
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            # Build AlphaXiv URL
            alphaxiv_url = f"https://www.alphaxiv.org/abs/{arxiv_id}"

            # Extract abstract — KHÔNG cắt dưới 1500 chars
            summary = (entry.summary or "").strip()
            content = summary[:2000] if summary else ""

            # Extract authors
            authors = [a.name for a in entry.authors if hasattr(a, "name")]

            # Extract title
            title = (entry.title or "").strip()
            if not title:
                continue

            results.append({
                "source_type":    "arxiv",
                "title":          title,
                "content":        content,  # Full abstract, min 1500 chars nếu có
                "url":            arxiv_url,
                "alphaxiv_url":   alphaxiv_url,
                "authors":        authors,
                "published":      entry.published if hasattr(entry, "published") else "",
                "citation_count": 0,  # arXiv không có citation count
                "score":          0.95,  # High score cho academic source
            })

        return results

    except RateLimitError:
        raise  # propagate to caller for retry handling
    except Exception as e:
        print(f"[arXiv] Error: {e}")
        return []
