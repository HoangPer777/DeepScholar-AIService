from types import SimpleNamespace


def test_vector_schema_status_reports_missing_v2_columns(monkeypatch):
    import app.embeddings.vector_store as vs

    fake_inspector = SimpleNamespace(
        get_table_names=lambda: ["article_chunks", "embeddings"],
        get_columns=lambda table: (
            [{"name": name} for name in ("id", "article_id", "chunk_index", "content")]
            if table == "article_chunks"
            else [{"name": name} for name in ("id", "chunk_id", "embedding", "created_at")]
        ),
    )
    monkeypatch.setattr(vs, "inspect", lambda _engine: fake_inspector)

    status = vs.vector_schema_status()

    assert status["status"] == "migration_required"
    assert "section" in status["missing_columns"]["article_chunks"]


def test_validate_vector_schema_accepts_complete_schema(monkeypatch):
    import app.embeddings.vector_store as vs

    fake_inspector = SimpleNamespace(
        get_table_names=lambda: ["article_chunks", "embeddings"],
        get_columns=lambda table: [
            {"name": name}
            for name in (
                vs.REQUIRED_ARTICLE_CHUNK_COLUMNS
                if table == "article_chunks"
                else vs.REQUIRED_EMBEDDING_COLUMNS
            )
        ],
    )
    monkeypatch.setattr(vs, "inspect", lambda _engine: fake_inspector)

    vs.validate_vector_schema()
