"""
Migration: Create sessions table for Deep Research Memory Chatbot

This migration creates the sessions table with:
- UUID primary key for session_id
- User isolation through user_id
- Indexes for efficient querying
- Row-level security policy for multi-user isolation

Requirements: 1.1, 1.2, 12.4
"""

from sqlalchemy import text
from app.core.database import engine, Base
from app.core.session_models import Session


def upgrade():
    """
    Create sessions table with indexes and row-level security.
    """
    print("Running migration: 001_create_sessions_table")
    
    try:
        with engine.begin() as conn:
            # Ensure UUID extension is available
            print("  - Ensuring uuid-ossp extension...")
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";'))
            
            # Create the sessions table using SQLAlchemy model
            print("  - Creating sessions table...")
            Base.metadata.create_all(bind=engine, tables=[Session.__table__])
            
            # Enable row-level security for multi-user isolation
            print("  - Enabling row-level security...")
            conn.execute(text("""
                ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
            """))
            
            # Create RLS policy for session isolation
            # This ensures users can only access their own sessions
            print("  - Creating row-level security policy...")
            conn.execute(text("""
                CREATE POLICY session_isolation ON sessions
                    FOR ALL
                    USING (user_id = current_setting('app.current_user_id', TRUE));
            """))
            
            print("✓ Migration completed successfully!")
            print("  - sessions table created")
            print("  - Indexes created: idx_user_sessions, idx_session_status")
            print("  - Row-level security enabled with session_isolation policy")
            
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        raise


def downgrade():
    """
    Drop sessions table and related objects.
    """
    print("Rolling back migration: 001_create_sessions_table")
    
    try:
        with engine.begin() as conn:
            # Drop the policy first
            print("  - Dropping row-level security policy...")
            conn.execute(text("""
                DROP POLICY IF EXISTS session_isolation ON sessions;
            """))
            
            # Drop the table
            print("  - Dropping sessions table...")
            conn.execute(text("""
                DROP TABLE IF EXISTS sessions CASCADE;
            """))
            
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
