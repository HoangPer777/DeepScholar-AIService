import re
import requests
from typing import Dict, List


def enrich_arxiv_metadata(sources: List[Dict]) -> List[Dict]:
    """
    For arxiv URLs, call Semantic Scholar API to get authors/year/venue.
    Non-arxiv sources are marked source_type = "web".
    """
    enriched = []
    for s in sources:
        url = s.get("url", "")
        arxiv_match = re.search(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d+)", url)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            try:
                api_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
                params = {"fields": "title,authors,year,venue,externalIds"}
                resp = requests.get(api_url, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    authors = [a["name"] for a in data.get("authors", [])]
                    s["apa_authors"] = authors
                    s["apa_year"] = data.get("year") or "n.d."
                    s["apa_venue"] = data.get("venue") or "arXiv preprint"
                    s["apa_title"] = data.get("title") or s["title"]
                    s["source_type"] = "arxiv"
                else:
                    s["source_type"] = "web"
            except Exception:
                s["source_type"] = "web"
        else:
            s["source_type"] = "web"
        enriched.append(s)
    return enriched


def format_apa_reference(index: int, source: Dict) -> str:
    """
    Build APA reference string from source dict.
    arxiv: Last, F. M., & Last, F. M. (Year). Title. Venue. URL
    web:   Title. (Year). domain. Retrieved from URL
    """
    url = source.get("url", "")
    title = source.get("apa_title", source.get("title", ""))
    year = source.get("apa_year", "n.d.")

    if source.get("source_type") == "arxiv":
        authors = source.get("apa_authors", [])
        venue = source.get("apa_venue", "arXiv preprint")

        if not authors:
            author_str = "Unknown Author"
        else:
            def fmt(name: str) -> str:
                parts = name.strip().split()
                if len(parts) < 2:
                    return name
                last = parts[-1]
                inits = " ".join(p[0] + "." for p in parts[:-1])
                return f"{last}, {inits}"

            formatted = [fmt(a) for a in authors]
            if len(formatted) == 1:
                author_str = formatted[0]
            elif len(formatted) <= 6:
                author_str = ", ".join(formatted[:-1]) + ", & " + formatted[-1]
            else:
                author_str = ", ".join(formatted[:6]) + ", et al."

        return f"[{index}] {author_str} ({year}). {title}. {venue}. {url}"
    else:
        domain = re.sub(r"https?://(www\.)?", "", url).split("/")[0]
        return f"[{index}] {title}. ({year}). {domain}. Retrieved from {url}"
