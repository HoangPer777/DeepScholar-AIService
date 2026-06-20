"""
Migration: Create messages table for Deep Research Memory Chatbot

This migration creates the messages table with:
- UUID primary key for message_id
- Foreign key to sessions(session_id) with ON DELETE CASCADE
- role, content, timestamp, token_count, and JSONB metadata columns
- Indexes for efficient session-scoped and time-based queries

Requirements: 2.1, 3.1
"""

from sqlalchemy import text
from app.core.database import engine, Base
from app.core.session_models import Message  # noqa: F401 – registers the model


def upgrade():
    """
    Create messages table with indexes and cascade delete constraint.
    """
    print("Running migration: 002_create_messages_table")

    try:
        with engine.begin() as conn:
            # Ensure UUID extension is available (idempotent)
            print("  - Ensuring uuid-ossp extension...")
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))

            # Create the messages table using the SQLAlchemy model definition.
            # Base.metadata.create_all is idempotent – it skips tables that
            # already exist, so it is safe to run multiple times.
            print("  - Creating messages table...")
            Base.metadata.create_all(
                bind=engine,
                tables=[Message.__table__],
                checkfirst=True,
            )

            print("✓ Migration completed successfully!")
            print("  - messages table created")
            print("  - Cascade delete constraint added (sessions → messages)")
            print("  - Indexes created: idx_session_messages, idx_message_timestamp")

    except Exception as e:
        print(f"✗ Migration failed: {e}")
        raise


def downgrade():
    """
    Drop messages table and related objects.
    """
    print("Rolling back migration: 002_create_messages_table")

    try:
        with engine.begin() as conn:
            print("  - Dropping messages table...")
            conn.execute(text("DROP TABLE IF EXISTS messages CASCADE;"))

            print("✓ Rollback completed successfully!")

    except Exception as e:
        print(f"✗ Rollback failed: {e}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "downgrade":
        downgrade()
    else:
        upgrade()
