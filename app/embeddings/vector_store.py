from sqlalchemy import inspect, text
from app.core.database import engine, SessionLocal
from app.embeddings.models import Base, ArticleChunk, Embedding
from app.embeddings.embedder import embed_texts

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
        print("pgvector extension ensured.")
    except Exception as e:
        # Extension may already exist or user may lack permissions — non-fatal
        print(f"Note: pgvector extension step: {e}")

    try:
        # Always attempt to create tables even if extension step had an issue
        Base.metadata.create_all(bind=engine)
        print("Database schema ensured (article_chunks + embeddings tables ready).")
    except Exception as e:
        print(f"Error creating schema tables: {e}")

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
