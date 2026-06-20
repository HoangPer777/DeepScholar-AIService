"""
Create the chunking v2 vector schema in the configured PostgreSQL/Supabase database.

This migration only resets the AI-service-owned tables:
  - embeddings
  - article_chunks

It refuses to delete non-empty tables unless --force is supplied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import inspect, text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import engine
from app.embeddings.models import ArticleChunk, Embedding
from app.embeddings.vector_store import validate_vector_schema


def _table_count(conn, table_name: str) -> int:
    if table_name not in inspect(conn).get_table_names():
        return 0
    return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


def upgrade(force: bool = False) -> None:
    with engine.begin() as conn:
        counts = {
            "article_chunks": _table_count(conn, "article_chunks"),
            "embeddings": _table_count(conn, "embeddings"),
        }
        if any(counts.values()) and not force:
            raise RuntimeError(
                f"Vector tables contain data: {counts}. "
                "Back up or re-ingest the data, then rerun with --force."
            )

        print(f"Resetting AI-service vector tables in configured database: {counts}")
        conn.execute(text("DROP TABLE IF EXISTS embeddings CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS article_chunks CASCADE"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        ArticleChunk.__table__.create(bind=conn, checkfirst=True)
        Embedding.__table__.create(bind=conn, checkfirst=True)

    validate_vector_schema()
    print("Vector schema v2 created and validated successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create DeepScholar vector schema v2.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow resetting non-empty article_chunks and embeddings tables.",
    )
    args = parser.parse_args()
    upgrade(force=args.force)


if __name__ == "__main__":
    main()
