from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reset_vector_db.py"


def test_reset_vector_db_only_drops_vector_tables():
    content = SCRIPT.read_text(encoding="utf-8")

    assert "DROP TABLE IF EXISTS embeddings" in content
    assert "DROP TABLE IF EXISTS article_chunks" in content
    assert "CREATE EXTENSION IF NOT EXISTS vector" in content


def test_reset_vector_db_does_not_drop_backend_tables():
    content = SCRIPT.read_text(encoding="utf-8").lower()
    forbidden_tables = [
        "drop table if exists articles",
        "drop table if exists users",
        "drop table if exists comments",
        "drop table if exists likes",
        "drop table if exists ranking",
    ]

    for forbidden in forbidden_tables:
        assert forbidden not in content
