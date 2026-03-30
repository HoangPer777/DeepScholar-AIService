from typing import List, Dict, Optional
import time

_store: List[Dict] = []


def memory_save(data: Dict) -> None:
    entry = {
        "q": data.get("q", ""),
        "a": data.get("a", ""),
        "timestamp": data.get("timestamp", time.time()),
    }
    _store.append(entry)
    if len(_store) > 100:
        _store.pop(0)


def memory_recall(query: Optional[str] = None, top_k: int = 3) -> List[Dict]:
    if not _store:
        return []

    if query is None:
        return [{"text": f"Q: {e['q']}\nA: {e['a']}", "source": "memory"} for e in _store[-top_k:]]

    query_words = set(query.lower().split())
    scored = []
    for entry in _store:
        entry_text = (entry["q"] + " " + entry["a"]).lower()
        overlap = sum(1 for w in query_words if w in entry_text)
        if overlap > 0:
            scored.append((overlap, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"text": f"Q: {e['q']}\nA: {e['a']}", "source": "memory"} for _, e in scored[:top_k]]


def memory_clear() -> None:
    _store.clear()