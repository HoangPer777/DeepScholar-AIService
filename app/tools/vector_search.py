from typing import Dict, List, Optional
import time

from app.embeddings.embedder import embed_texts
from app.embeddings.vector_store import similarity_search


def search_article_chunks(
    article_id: int,
    question: str,
    focus_sections: Optional[List[str]] = None,
    limit: int = 8,
    timings: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    """
    Embed question and run PGVector similarity search filtered by article_id.
    Returns top-N chunks as list of dicts with chunk_id, content, distance.
    """
    retrieval_start = time.perf_counter()

    t0 = time.perf_counter()
    query_embedding = embed_texts([question])[0]
    if timings is not None:
        timings["embedding_ms"] = int((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    results = similarity_search(
        article_id,
        query_embedding,
        limit=limit,
        focus_sections=focus_sections,
    )
    if timings is not None:
        timings["vector_search_ms"] = int((time.perf_counter() - t0) * 1000)
        timings["total_retrieval_ms"] = int((time.perf_counter() - retrieval_start) * 1000)
        timings["retrieval_ms"] = timings["total_retrieval_ms"]
    return results
