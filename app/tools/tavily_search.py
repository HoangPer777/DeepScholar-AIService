from typing import Dict, List

from tavily import TavilyClient

from app.core.config import settings


def tavily_search(query: str, max_results: int = 10) -> List[Dict]:
    """Search web using Tavily API. Returns list of result dicts."""
    try:
        tavily = TavilyClient(api_key=settings.TAVILY_API_KEY)
        res = tavily.search(query=query, max_results=max_results)
        return [
            {
                "title":   r.get("title", ""),
                "content": r.get("content", "")[:700],
                "url":     r.get("url", ""),
                "score":   round(r.get("score", 0.0), 3),
            }
            for r in res.get("results", [])
        ]
    except Exception as e:
        return [{"title": "Error", "content": str(e), "url": "", "score": 0.0}]
