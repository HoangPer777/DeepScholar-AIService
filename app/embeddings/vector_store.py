import traceback
from sqlalchemy import inspect, text
from app.core.database import engine, SessionLocal
from app.embeddings.models import Base, ArticleChunk, Embedding
from app.embeddings.embedder import embed_texts
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

def database_health() -> dict:
    """
    Check database connection and pgvector extension status
    """
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            return {"status": "ok", "message": "Connected to PostgreSQL"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def ensure_pgvector_schema():
    """
    Create pgvector extension and embeddings table if not exists.
    SQLAlchemy 2.x requires raw SQL to be wrapped in text().
    """
    try:
        with engine.begin() as conn:
            # text() is required in SQLAlchemy 2.x for raw SQL strings
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension ensured.")
    except Exception as e:
        logger.error(
            "pgvector extension is not available. Enable it in the Supabase dashboard under Database > Extensions."
        )
        raise

    try:
        # Always attempt to create tables even if extension step had an issue
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema ensured (article_chunks + embeddings tables ready).")
    except Exception as e:
        logger.error(f"Error creating schema tables: {e}")
        raise

def ingest_article_chunks(article_id: int, chunks: list[str]) -> dict:
    """
    Store article chunks and embeddings in database.
    Deletes old chunks, creates new chunks, generates and stores embeddings.
    """
    if not chunks:
        return {"stored": False, "chunk_count": 0, "message": "No chunks provided"}

    ensure_pgvector_schema()
    
    with SessionLocal() as session:
        try:
            # 1. Delete old chunks for this article
            old_chunks = session.query(ArticleChunk).filter(ArticleChunk.article_id == article_id).all()
            if old_chunks:
                for chunk in old_chunks:
                    session.delete(chunk)
                session.commit()

            # 2. Generate embeddings for all chunks in batch
            embeddings_list = embed_texts(chunks)

            # 2a. Validate embedding dimension before attempting pgvector insert
            if embeddings_list and len(embeddings_list[0]) != settings.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Dimension mismatch: model returned {len(embeddings_list[0])}D but EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION}"
                )

            # 3. Insert new chunks and their embeddings
            for i, (chunk_text, embedding_vector) in enumerate(zip(chunks, embeddings_list)):
                db_chunk = ArticleChunk(
                    article_id=article_id,
                    chunk_index=i,
                    content=chunk_text
                )
                session.add(db_chunk)
                session.flush() # Flush to get chunk ID
                
                db_embedding = Embedding(
                    chunk_id=db_chunk.id,
                    embedding=embedding_vector
                )
                session.add(db_embedding)

            session.commit()
            return {"stored": True, "chunk_count": len(chunks)}
        except Exception as e:
            session.rollback()
            logger.error(f"ingest_article_chunks failed: {e}\n{traceback.format_exc()}")
            return {"stored": False, "chunk_count": 0, "error": str(e)}

def similarity_search(article_id: int, query_embedding: list[float], limit: int = 5):
    """
    Query for similar chunks using vector similarity (<=> operator in pgvector means L2 distance or Cosine distance)
    """
    # Assuming L2 distance
    with SessionLocal() as session:
        results = session.query(
            ArticleChunk, 
            Embedding.embedding.l2_distance(query_embedding).label('distance')
        ).join(Embedding)\
         .filter(ArticleChunk.article_id == article_id)\
         .order_by('distance')\
         .limit(limit)\
         .all()
        
        return [
            {
                "chunk_id": row.ArticleChunk.id,
                "content": row.ArticleChunk.content,
                "distance": row.distance
            } for row in results
        ]
