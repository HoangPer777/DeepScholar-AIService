import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("INTERNAL_SERVICE_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "google")
os.environ.setdefault("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-2-preview")
os.environ.setdefault("EMBEDDING_DIMENSION", "3072")

from app.embeddings.models import ArticleChunk, Embedding
from app.pdf_pipeline.chunker import PaperChunk


def _paper_chunk(index=0, section="methodology", chunk_type="section_text"):
    return PaperChunk(
        content=f"chunk {index}",
        content_for_embedding=f"Title: T\nSection: {section}\n\nchunk {index}",
        chunk_index=index,
        chunk_type=chunk_type,
        section=section,
        section_title=section.title(),
        section_level=2,
        heading_path=["T", section.title()],
        token_count=10,
        metadata={"sample": True},
        chunking_version="v2",
    )


def test_article_chunk_schema_contains_v2_metadata_columns():
    columns = set(ArticleChunk.__table__.columns.keys())

    assert {
        "article_id",
        "chunk_index",
        "content",
        "section",
        "section_title",
        "chunk_type",
        "heading_path",
        "page_start",
        "page_end",
        "token_count",
        "metadata",
        "chunking_version",
        "created_at",
    }.issubset(columns)
    assert Embedding.__table__.columns["embedding"] is not None
    assert ArticleChunk.__table__.columns["section_title"].type.__class__.__name__ == "Text"


class FakeQuery:
    def __init__(self, old_chunks):
        self.old_chunks = old_chunks

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self.old_chunks)

    def delete(self, synchronize_session=False):
        count = len(self.old_chunks)
        self.old_chunks.clear()
        return count


class FakeSession:
    def __init__(self):
        self.deleted = []
        self.added = []
        self.committed = False
        self.rolled_back = False
        self.old_chunks = []
        self._next_id = 100

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, *_args, **_kwargs):
        return FakeQuery(self.old_chunks)

    def delete(self, obj):
        self.deleted.append(obj)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if isinstance(obj, ArticleChunk) and obj.id is None:
                obj.id = self._next_id
                self._next_id += 1

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_ingest_paper_chunks_inserts_metadata(monkeypatch):
    import app.embeddings.vector_store as vs

    fake_session = FakeSession()
    monkeypatch.setattr(vs, "ensure_pgvector_schema", lambda: None)
    monkeypatch.setattr(vs, "SessionLocal", lambda: fake_session)
    monkeypatch.setattr(vs, "embed_texts", lambda texts: [[0.1] * 3072 for _ in texts])

    result = vs.ingest_paper_chunks(42, [_paper_chunk(0), _paper_chunk(1, "results", "table")])

    added_chunks = [obj for obj in fake_session.added if isinstance(obj, ArticleChunk)]
    added_embeddings = [obj for obj in fake_session.added if isinstance(obj, Embedding)]
    assert result["stored"] is True
    assert result["chunk_count"] == 2
    assert result["chunk_type_counts"] == {"section_text": 1, "table": 1}
    assert len(added_chunks) == 2
    assert len(added_embeddings) == 2
    assert added_chunks[0].section == "methodology"
    assert added_chunks[0].metadata_json == {"sample": True}
    assert fake_session.committed is True


def test_ingest_paper_chunks_rejects_wrong_dimension(monkeypatch):
    import app.embeddings.vector_store as vs

    fake_session = FakeSession()
    monkeypatch.setattr(vs, "ensure_pgvector_schema", lambda: None)
    monkeypatch.setattr(vs, "SessionLocal", lambda: fake_session)
    monkeypatch.setattr(vs, "embed_texts", lambda texts: [[0.1] * 128 for _ in texts])

    result = vs.ingest_paper_chunks(42, [_paper_chunk(0)])

    assert result["stored"] is False
    assert "Dimension mismatch" in result["error"]
    assert fake_session.rolled_back is True


def test_ingest_paper_chunks_rejects_missing_embeddings(monkeypatch):
    import app.embeddings.vector_store as vs

    fake_session = FakeSession()
    monkeypatch.setattr(vs, "ensure_pgvector_schema", lambda: None)
    monkeypatch.setattr(vs, "SessionLocal", lambda: fake_session)
    monkeypatch.setattr(vs, "embed_texts", lambda texts: [[0.1] * 3072])

    result = vs.ingest_paper_chunks(42, [_paper_chunk(0), _paper_chunk(1)])

    assert result["stored"] is False
    assert "Embedding count mismatch" in result["error"]
    assert fake_session.rolled_back is True


def _row(chunk_id, section, distance):
    chunk = SimpleNamespace(
        id=chunk_id,
        content=f"content {chunk_id}",
        section=section,
        section_title=section.title(),
        chunk_type="section_text",
        heading_path='["Title", "Section"]',
        page_start=None,
        page_end=None,
        chunk_index=chunk_id,
        metadata_json={},
    )
    return SimpleNamespace(ArticleChunk=chunk, distance=distance)


def test_similarity_search_prefers_focus_sections(monkeypatch):
    import app.embeddings.vector_store as vs

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_run(_session, article_id, query_embedding, limit, sections=None):
        assert article_id == 7
        if sections:
            return [_row(1, "methodology", 0.1), _row(2, "methodology", 0.2), _row(3, "methodology", 0.3)]
        return [_row(9, "results", 0.4)]

    monkeypatch.setattr(vs, "SessionLocal", lambda: DummySession())
    monkeypatch.setattr(vs, "_run_similarity_query", fake_run)

    results = vs.similarity_search(7, [0.1] * 3072, limit=3, focus_sections=["methodology"])

    assert [r["section"] for r in results] == ["methodology", "methodology", "methodology"]
    assert {"content", "chunk_id", "distance", "heading_path"}.issubset(results[0].keys())


def test_similarity_search_falls_back_when_section_results_are_too_few(monkeypatch):
    import app.embeddings.vector_store as vs

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_run(_session, article_id, query_embedding, limit, sections=None):
        if sections:
            return [_row(1, "methodology", 0.1)]
        return [_row(1, "methodology", 0.1), _row(2, "results", 0.2), _row(3, "discussion", 0.3)]

    monkeypatch.setattr(vs, "SessionLocal", lambda: DummySession())
    monkeypatch.setattr(vs, "_run_similarity_query", fake_run)

    results = vs.similarity_search(7, [0.1] * 3072, limit=3, focus_sections=["methodology"], min_results=2)

    assert [r["chunk_id"] for r in results] == [1, 2, 3]
