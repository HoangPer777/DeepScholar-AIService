import json
import math
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List


def external_search(query: str, top_k: int = 5) -> List[Dict]:
    use_mock = os.getenv("USE_MOCK_EXTERNAL", "false").lower() == "true"
    google_key = os.getenv("GOOGLE_API_KEY", "")
    google_cse = os.getenv("GOOGLE_CSE_ID", "")

    if use_mock:
        return _mock_external_search(query, top_k)

    results: List[Dict] = []

    if google_key and google_cse:
        try:
            results.extend(_google_search(query, top_k=min(top_k, 10)))
        except Exception as exc:
            print(f"[tool:external_search] google search failed: {exc}", flush=True)
    else:
        print("[tool:external_search] skip google search: GOOGLE_API_KEY/GOOGLE_CSE_ID missing", flush=True)

    if os.getenv("ENABLE_SEMANTIC_SCHOLAR", "true").lower() == "true":
        try:
            results.extend(_semantic_scholar_search(query, top_k=3))
        except Exception as exc:
            print(f"[tool:external_search] semantic scholar failed: {exc}", flush=True)

    if not results:
        try:
            results.extend(_arxiv_search(query, top_k=min(top_k, 5)))
        except Exception as exc:
            print(f"[tool:external_search] arxiv fallback failed: {exc}", flush=True)

    if not results:
        raise RuntimeError(
            "No external results available. Configure GOOGLE_CSE_ID or Semantic Scholar API/network, "
            "or set USE_MOCK_EXTERNAL=true for demo mode."
        )

    return results[:top_k]


def _google_search(query: str, top_k: int = 5) -> List[Dict]:
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    params = urllib.parse.urlencode(
        {
            "key": api_key,
            "cx": cse_id,
            "q": query,
            "num": min(top_k, 10),
            "safe": "active",
            "fields": "items(title,link,snippet,pagemap/metatags)",
        }
    )
    url = f"https://customsearch.googleapis.com/customsearch/v1?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results: List[Dict] = []
    for i, item in enumerate(data.get("items", [])):
        metatags = item.get("pagemap", {}).get("metatags", [{}])[0]
        description = metatags.get("og:description", "") or item.get("snippet", "")
        results.append(
            {
                "text": f"{item.get('title', '')}\n{description}",
                "source": "google_search",
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "score": round(1.0 - (i * 0.05), 2),
            }
        )
    return results


def _semantic_scholar_search(query: str, top_k: int = 3) -> List[Dict]:
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,abstract,authors,year,citationCount,externalIds,openAccessPdf"
    params = urllib.parse.urlencode({"query": query, "limit": top_k, "fields": fields})

    headers = {"Accept": "application/json"}
    scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    if scholar_key:
        headers["x-api-key"] = scholar_key

    req = urllib.request.Request(f"{base_url}?{params}", headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results: List[Dict] = []
    for paper in data.get("data", []):
        title = paper.get("title", "")
        abstract = paper.get("abstract") or "(No abstract)"
        year = paper.get("year", "")
        n_cite = paper.get("citationCount", 0)
        doi = paper.get("externalIds", {}).get("DOI", "")
        authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:3])
        pdf_url = (paper.get("openAccessPdf") or {}).get("url", "")
        url = pdf_url or (f"https://doi.org/{doi}" if doi else "")
        citation_score = min(1.0, math.log10(max(n_cite, 1) + 1) / 4)

        results.append(
            {
                "text": f"Title: {title}\nAuthors: {authors} ({year})\nCitations: {n_cite}\nAbstract: {abstract[:600]}",
                "source": "semantic_scholar",
                "title": title,
                "authors": authors,
                "year": str(year),
                "doi": doi,
                "url": url,
                "score": round(citation_score, 4),
            }
        )
    return results


def _arxiv_search(query: str, top_k: int = 3) -> List[Dict]:
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": min(top_k, 10),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"http://export.arxiv.org/api/query?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/atom+xml"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        xml_data = resp.read().decode("utf-8", errors="ignore")

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    results: List[Dict] = []
    for i, entry in enumerate(root.findall("atom:entry", ns)):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:4]
        authors = ", ".join(
            (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
            for author in entry.findall("atom:author", ns)[:3]
        )

        link = ""
        for ln in entry.findall("atom:link", ns):
            if ln.attrib.get("type") == "application/pdf":
                link = ln.attrib.get("href", "")
                break
        if not link:
            link = entry.findtext("atom:id", default="", namespaces=ns) or ""

        results.append(
            {
                "text": f"Title: {title}\nAuthors: {authors} ({published})\nAbstract: {summary[:600]}",
                "source": "arxiv",
                "title": title,
                "authors": authors,
                "year": published,
                "doi": "",
                "url": link,
                "score": round(1.0 - (i * 0.06), 2),
            }
        )
    return results


def _mock_external_search(query: str, top_k: int) -> List[Dict]:
    return [
        {
            "text": f"[MOCK] Google Search result for '{query}'",
            "source": "mock_google",
            "title": f"Mock Web Result: {query}",
            "url": "https://example.com/mock-google",
            "score": 0.88,
        },
        {
            "text": f"[MOCK] Semantic Scholar result for '{query}'",
            "source": "mock_semantic_scholar",
            "title": f"Mock Academic Paper: {query}",
            "authors": "Nguyen et al.",
            "year": "2024",
            "doi": "10.xxxx/mock",
            "url": "https://arxiv.org/mock",
            "score": 0.82,
        },
    ][:top_k]


def deep_research(query: str) -> List[Dict]:
    return external_search(query=query, top_k=5)
