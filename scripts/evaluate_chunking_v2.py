from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pdf_pipeline.chunker import chunk_paper


DEFAULT_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "papers"


def _tokens(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}").lower()
        for token in text.split()
        if len(token.strip(".,:;!?()[]{}")) > 2
    }


def _rank_chunks(question: str, chunks):
    query_tokens = _tokens(question)
    ranked = []
    for chunk in chunks:
        chunk_tokens = _tokens(chunk.content_for_embedding)
        overlap = len(query_tokens & chunk_tokens)
        ranked.append((overlap, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _score, chunk in ranked]


def evaluate(fixture_dir: Path = DEFAULT_FIXTURE_DIR) -> dict:
    markdown = (fixture_dir / "llamaparse_sample_ieee.md").read_text(encoding="utf-8")
    questions = json.loads((fixture_dir / "evaluation_questions.json").read_text(encoding="utf-8"))

    started = time.perf_counter()
    chunks = chunk_paper(
        title="Structure-Aware RAG",
        abstract="This paper proposes structure-aware chunking for scientific retrieval.",
        content=markdown,
    )
    chunking_ms = int((time.perf_counter() - started) * 1000)

    precision_values = []
    recall_values = []
    section_hits = []
    table_hits = []

    for question in questions:
        ranked = _rank_chunks(question["question"], chunks)
        top5 = ranked[:5]
        top8 = ranked[:8]
        expected_section = question.get("expected_section")
        expected_type = question.get("expected_chunk_type")

        def is_relevant(chunk):
            if expected_type and chunk.chunk_type == expected_type:
                return True
            return chunk.section == expected_section

        precision_values.append(sum(1 for chunk in top5 if is_relevant(chunk)) / max(1, len(top5)))
        recall_values.append(1.0 if any(is_relevant(chunk) for chunk in top8) else 0.0)
        section_hits.append(1.0 if top5 and top5[0].section == expected_section else 0.0)
        if expected_type == "table":
            table_hits.append(1.0 if any(chunk.chunk_type == "table" for chunk in top8) else 0.0)

    token_counts = [chunk.token_count for chunk in chunks]
    type_counts = {}
    for chunk in chunks:
        type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1

    p95_index = max(0, int(len(token_counts) * 0.95) - 1)
    sorted_tokens = sorted(token_counts)

    return {
        "chunk_count": len(chunks),
        "chunk_type_counts": type_counts,
        "avg_chunk_tokens": round(statistics.mean(token_counts), 2) if token_counts else 0,
        "p95_chunk_tokens": sorted_tokens[p95_index] if sorted_tokens else 0,
        "precision@5": round(statistics.mean(precision_values), 3) if precision_values else 0,
        "recall@8": round(statistics.mean(recall_values), 3) if recall_values else 0,
        "section_hit_rate": round(statistics.mean(section_hits), 3) if section_hits else 0,
        "table_answer_hit_rate": round(statistics.mean(table_hits), 3) if table_hits else 0,
        "chunking_latency_ms": chunking_ms,
        "google_embedding_requests_per_paper": len(chunks),
    }


def main() -> None:
    fixture_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FIXTURE_DIR
    report = evaluate(fixture_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
