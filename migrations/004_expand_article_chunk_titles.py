"""
Allow full scientific section and caption titles in article_chunks.

This is an additive migration and does not delete vector data.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import engine


def upgrade() -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                ALTER TABLE article_chunks
                ALTER COLUMN section_title TYPE TEXT
                """
            )
        )
    print("article_chunks.section_title migrated to TEXT.")


if __name__ == "__main__":
    upgrade()
