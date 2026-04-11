import json
import re
from typing import TYPE_CHECKING


def safe_json(text: str) -> dict:
    """Parse JSON an toàn từ LLM output (có thể có markdown fences)."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            try:
                return json.loads(m.group().replace("'", '"'))
            except Exception:
                pass
    print(f"  [WARN] safe_json failed:\n{text[:300]}")
    return {}


def log(state, msg: str):
    """Append log message to state.logs và print."""
    state.logs.append(msg)
    print(msg)
    return state


def effective_question(state) -> str:
    """Trả về clarified_question nếu có, fallback về question gốc."""
    return getattr(state, "clarified_question", None) or state.question
