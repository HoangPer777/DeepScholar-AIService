from app.core.database import engine
from sqlalchemy import text

def reset_vector_db():
    print("Resetting Vector DB tables due to dimension mismatch...")
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS embeddings CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS article_chunks CASCADE;"))
        print("Success: Tables dropped. They will be recreated on the next request with 3072 dimensions.")
    except Exception as e:
        print(f"Error resetting DB: {e}")

if __name__ == "__main__":
    reset_vector_db()
