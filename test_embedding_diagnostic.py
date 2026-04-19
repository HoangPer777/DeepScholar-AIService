"""
Diagnostic script: test embedding pipeline end-to-end
Run from DeepScholar-AIService directory:
    python test_embedding_diagnostic.py
"""
import sys
import os

# Load .env
from dotenv import load_dotenv
load_dotenv(".env", override=True)

print("=" * 60)
print("STEP 1: ENV CHECK")
print("=" * 60)
db_url = os.getenv("DATABASE_URL", "NOT SET")
print(f"DATABASE_URL: {db_url[:60]}...")
print(f"EMBEDDING_PROVIDER: {os.getenv('EMBEDDING_PROVIDER')}")
print(f"EMBEDDING_DIMENSION: {os.getenv('EMBEDDING_DIMENSION')}")
print(f"GOOGLE_API_KEY set: {bool(os.getenv('GOOGLE_API_KEY'))}")
print(f"GOOGLE_EMBEDDING_MODEL: {os.getenv('GOOGLE_EMBEDDING_MODEL', 'NOT SET (will use default)')}")
print(f"INTERNAL_SERVICE_KEY set: {bool(os.getenv('INTERNAL_SERVICE_KEY'))}")

print("\n" + "=" * 60)
print("STEP 2: CONFIG LOAD")
print("=" * 60)
try:
    from app.core.config import settings
    print(f"settings.EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")
    print(f"settings.EMBEDDING_DIMENSION: {settings.EMBEDDING_DIMENSION}")
    print(f"settings.DATABASE_URL: {settings.DATABASE_URL[:60]}...")
    print("CONFIG OK")
except Exception as e:
    print(f"CONFIG ERROR: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 3: DATABASE CONNECTION")
print("=" * 60)
try:
    from app.core.database import engine
    from sqlalchemy import text
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        print(f"DB connected: {result.fetchone()[0][:50]}")

    # Check pgvector extension
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'"
        ))
        row = result.fetchone()
        if row:
            print(f"pgvector extension: installed (version {row[0]})")
        else:
            print("pgvector extension: NOT FOUND - enable in Supabase Dashboard > Database > Extensions")
            sys.exit(1)
except Exception as e:
    print(f"DB ERROR: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 4: TABLE CHECK")
print("=" * 60)
try:
    with engine.connect() as conn:
        # Check article_chunks
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'article_chunks')"
        ))
        chunks_exists = result.fetchone()[0]
        print(f"article_chunks table exists: {chunks_exists}")

        # Check embeddings
        result = conn.execute(text(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'embeddings')"
        ))
        emb_exists = result.fetchone()[0]
        print(f"embeddings table exists: {emb_exists}")

        if emb_exists:
            # Check vector dimension
            result = conn.execute(text("""
                SELECT udt_name, character_maximum_length
                FROM information_schema.columns
                WHERE table_name = 'embeddings' AND column_name = 'embedding'
            """))
            row = result.fetchone()
            print(f"embeddings.embedding column type: {row}")

        # Row counts
        if chunks_exists:
            result = conn.execute(text("SELECT COUNT(*) FROM article_chunks"))
            print(f"article_chunks row count: {result.fetchone()[0]}")
        if emb_exists:
            result = conn.execute(text("SELECT COUNT(*) FROM embeddings"))
            print(f"embeddings row count: {result.fetchone()[0]}")
except Exception as e:
    print(f"TABLE CHECK ERROR: {e}")

print("\n" + "=" * 60)
print("STEP 5: EMBEDDER TEST (single text)")
print("=" * 60)
try:
    from app.embeddings.embedder import embed_texts
    test_texts = ["This is a test sentence for embedding."]
    print(f"Calling embed_texts with provider={settings.EMBEDDING_PROVIDER}...")
    result = embed_texts(test_texts)
    if result and len(result) > 0:
        vec = result[0]
        print(f"Embedding generated: dim={len(vec)}, first 5 values={vec[:5]}")
        if len(vec) != settings.EMBEDDING_DIMENSION:
            print(f"WARNING: dim mismatch! Got {len(vec)}, expected {settings.EMBEDDING_DIMENSION}")
            print("Fix: update EMBEDDING_DIMENSION in .env to match actual model output")
        else:
            print("Dimension matches config OK")
    else:
        print("ERROR: embed_texts returned empty result")
except Exception as e:
    print(f"EMBEDDER ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("STEP 6: FULL INGEST TEST (article_id=99999)")
print("=" * 60)
try:
    from app.embeddings.vector_store import ingest_article_chunks
    test_chunks = [
        "This is chunk 1 of a test article about machine learning.",
        "This is chunk 2 discussing neural networks and deep learning.",
    ]
    print("Calling ingest_article_chunks(article_id=99999, 2 chunks)...")
    result = ingest_article_chunks(99999, test_chunks)
    print(f"Result: {result}")

    if result.get("stored"):
        print("INGEST SUCCESS - cleaning up test data...")
        from app.core.database import SessionLocal
        from app.embeddings.models import ArticleChunk
        with SessionLocal() as session:
            deleted = session.query(ArticleChunk).filter(ArticleChunk.article_id == 99999).delete()
            session.commit()
            print(f"Cleaned up {deleted} test chunks")
    else:
        print(f"INGEST FAILED: {result.get('error', 'unknown error')}")
except Exception as e:
    print(f"INGEST ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
