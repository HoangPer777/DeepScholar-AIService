# TODO: Extract metadata from PDF/text


def infer_metadata(text: str, fallback_title: str) -> dict[str, str]:
    """
    TODO: Extract title, authors, abstract from PDF text
    Use heuristics or LLM for better parsing
    """
    # TODO: Implementation
    return {
        "title": fallback_title,
        "authors": "",
        "abstract": "",
        "content": text,
    }
