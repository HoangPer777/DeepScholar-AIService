import re
import requests
from typing import Dict, List, Tuple


MAX_CITABLE_SOURCES = 16


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


def select_citable_sources(
    sources: List[Dict],
    limit: int = MAX_CITABLE_SOURCES,
) -> List[Tuple[int, Dict]]:
    """Return a bounded, evidence-bearing source set with stable public indexes."""
    selected: List[Tuple[int, Dict]] = []
    real_index = 0
    for source in sources:
        if source.get("title") == "__research_notes__":
            continue
        real_index += 1
        if not (source.get("content") or "").strip():
            continue
        selected.append((real_index, source))
        if len(selected) >= limit:
            break
    return selected


def split_reference_section(draft: str) -> Tuple[str, str]:
    """Split a Markdown draft into body and References content."""
    parts = re.split(r"(?im)^\s*#+\s*references\s*$", draft or "", maxsplit=1)
    return parts[0].rstrip(), parts[1].strip() if len(parts) == 2 else ""


def canonicalize_references(draft: str, references: Dict[int, str]) -> str:
    """Replace model-written references with canonical entries for cited sources."""
    body, _ = split_reference_section(draft)
    cited = sorted({int(marker) for marker in re.findall(r"\[(\d+)\]", body)})
    entries = [references[index] for index in cited if index in references]
    if not entries:
        return body
    return f"{body}\n\n## References\n" + "\n".join(entries)
