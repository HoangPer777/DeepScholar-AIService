import re
from typing import Dict, List, Tuple


def build_citation_map(ranked_context: List[Dict]) -> Dict[int, Dict]:
    return {i: c for i, c in enumerate(ranked_context, 1)}


def extract_citations(answer: str) -> List[int]:
    raw = re.findall(r"\[(\d+)\]", answer or "")
    return [int(x) for x in raw]


def validate_citations(answer: str, ranked_context: List[Dict]) -> Tuple[bool, str]:
    refs = extract_citations(answer)
    if not refs:
        return False, "Missing citations [n] in final answer"

    max_ref = len(ranked_context)
    invalid = [r for r in refs if r < 1 or r > max_ref]
    if invalid:
        return False, f"Invalid citations: {invalid}; max allowed is [{max_ref}]"

    return True, "ok"


def append_reference_list(answer: str, ranked_context: List[Dict]) -> str:
    lines = [answer.strip(), "", "References:"]
    for i, ctx in enumerate(ranked_context, 1):
        title = ctx.get("title") or "Untitled"
        source = ctx.get("source") or "unknown"
        url = ctx.get("url") or ""
        doi = ctx.get("doi") or ""
        tail = f" | DOI: {doi}" if doi else ""
        tail += f" | URL: {url}" if url else ""
        lines.append(f"[{i}] {title} ({source}){tail}")
    return "\n".join(lines)
