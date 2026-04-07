from typing import Dict, List, Optional

from app.embeddings.embedder import embed_texts
from app.embeddings.vector_store import similarity_search


def search_article_chunks(
    article_id: int,
    question: str,
    focus_sections: Optional[List[str]] = None,
    limit: int = 8,
) -> List[Dict]:
    """
    Embed question and run PGVector similarity search filtered by article_id.
    Returns top-N chunks as list of dicts with chunk_id, content, distance.
    """
    query_embedding = embed_texts([question])[0]
    results = similarity_search(article_id, query_embedding, limit=limit)
    return results
