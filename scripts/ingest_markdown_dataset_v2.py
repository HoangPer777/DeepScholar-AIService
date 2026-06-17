from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings.vector_store import ingest_paper_chunks
from app.pdf_pipeline.chunker import chunk_paper
from scripts.reset_vector_db import reset_vector_db


def load_manifest(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_manifest(manifest_path: Path, dry_run: bool = False, reset: bool = False) -> list[dict]:
    records = load_manifest(manifest_path)
    base_dir = manifest_path.parent
    reports = []

    if reset and not dry_run:
        reset_vector_db()

    for record in records:
        content_path = base_dir / record["content_file"]
        content = content_path.read_text(encoding="utf-8")
        chunks = chunk_paper(
            title=record["title"],
            abstract=record.get("abstract", ""),
            content=content,
            source_format="llamaparse_markdown",
        )
        type_counts = {}
        for chunk in chunks:
            type_counts[chunk.chunk_type] = type_counts.get(chunk.chunk_type, 0) + 1

        if dry_run:
            result = {
                "stored": False,
                "dry_run": True,
                "chunk_count": len(chunks),
                "chunk_type_counts": type_counts,
            }
        else:
            result = ingest_paper_chunks(int(record["article_id"]), chunks)

        reports.append(
            {
                "article_id": record["article_id"],
                "title": record["title"],
                **result,
            }
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Markdown paper dataset with chunking v2.")
    parser.add_argument("manifest", type=Path, help="Path to dataset_manifest.json")
    parser.add_argument("--dry-run", action="store_true", help="Chunk only; do not call DB or Google embedding.")
    parser.add_argument("--reset-vector-db", action="store_true", help="Drop embeddings/article_chunks before ingest.")
    args = parser.parse_args()

    report = ingest_manifest(args.manifest, dry_run=args.dry_run, reset=args.reset_vector_db)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
