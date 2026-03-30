import os
from typing import List, Dict


def rerank(question: str, contexts: List[Dict], top_k: int = 5) -> List[Dict]:
    if not contexts:
        return []

    use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
    if use_reranker:
        try:
            return _cross_encoder_rerank(question, contexts, top_k)
        except Exception:
            pass
    return _score_based_rerank(contexts, top_k)


def _cross_encoder_rerank(question: str, contexts: List[Dict], top_k: int) -> List[Dict]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    pairs = [(question, c.get("text", "")) for c in contexts]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, contexts), key=lambda x: x[0], reverse=True)
    output = []
    for score, ctx in ranked[:top_k]:
        item = dict(ctx)
        item["rerank_score"] = round(float(score), 4)
        output.append(item)
    return output


def _score_based_rerank(contexts: List[Dict], top_k: int) -> List[Dict]:
    seen = set()
    unique = []
    for c in contexts:
        text = c.get("text", "")
        if text not in seen:
            seen.add(text)
            unique.append(c)
    unique.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return unique[:top_k]