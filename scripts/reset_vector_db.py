from sqlalchemy import text

from app.core.database import engine
from app.embeddings.models import ArticleChunk, Embedding
from app.embeddings.vector_store import validate_vector_schema


ALLOWED_DROP_TABLES = ("embeddings", "article_chunks")


def reset_vector_db() -> None:
    """
    Drop only AI-service vector tables. Do not touch Django backend tables.
    """
    print("Resetting AI-service vector tables: embeddings, article_chunks")
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS embeddings CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS article_chunks CASCADE;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        ArticleChunk.__table__.create(bind=conn, checkfirst=True)
        Embedding.__table__.create(bind=conn, checkfirst=True)
    validate_vector_schema()
    print("Vector tables recreated and validated with schema v2.")


if __name__ == "__main__":
    reset_vector_db()
