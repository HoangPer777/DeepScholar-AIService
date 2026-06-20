import json
import traceback
from collections import Counter
from typing import Iterable, Optional

from sqlalchemy import inspect, text

from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.core.logger import get_logger
from app.embeddings.embedder import embed_texts
from app.embeddings.models import ArticleChunk, Base, Embedding
from app.pdf_pipeline.chunker import PaperChunk, normalize_section_name

logger = get_logger(__name__)

REQUIRED_ARTICLE_CHUNK_COLUMNS = {
    "id",
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
}
REQUIRED_EMBEDDING_COLUMNS = {"id", "chunk_id", "embedding", "created_at"}


class VectorSchemaMismatchError(RuntimeError):
    pass


def vector_schema_status() -> dict:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    article_columns = (
        {column["name"] for column in inspector.get_columns("article_chunks")}
        if "article_chunks" in table_names
        else set()
    )
    embedding_columns = (
        {column["name"] for column in inspector.get_columns("embeddings")}
        if "embeddings" in table_names
        else set()
    )
    missing = {
        "article_chunks": sorted(REQUIRED_ARTICLE_CHUNK_COLUMNS - article_columns),
        "embeddings": sorted(REQUIRED_EMBEDDING_COLUMNS - embedding_columns),
    }
    ready = not missing["article_chunks"] and not missing["embeddings"]
    return {
        "status": "ready" if ready else "migration_required",
        "version": settings.CHUNKING_VERSION if ready else "legacy_or_missing",
        "missing_columns": missing,
    }


def validate_vector_schema() -> None:
    status = vector_schema_status()
    if status["status"] != "ready":
        raise VectorSchemaMismatchError(
            "Supabase vector schema is not compatible with chunking v2. "
            f"Missing columns: {status['missing_columns']}. "
            "Run: python migrations/003_create_vector_schema_v2.py"
        )


def database_health() -> dict:
    """
    Check database connection and pgvector extension status.
    """
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            return {
                "status": "ok",
                "message": "Connected to PostgreSQL",
                "vector_schema": vector_schema_status(),
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def ensure_pgvector_schema():
    """
    Create pgvector extension and v2 embeddings schema if missing.
    Existing v1 tables should be reset before this migration path in dev/staging.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension ensured.")
    except Exception:
        logger.error(
            "pgvector extension is not available. Enable it in the Supabase dashboard under Database > Extensions."
        )
        raise

    try:
        Base.metadata.create_all(bind=engine)
        validate_vector_schema()
        logger.info("Database schema validated (article_chunks v2 + embeddings ready).")
    except Exception as e:
        logger.error(f"Error creating schema tables: {e}")
        raise


def _validate_embedding_dimensions(embeddings_list: list[list[float]]) -> None:
    for index, embedding in enumerate(embeddings_list):
        if len(embedding) != settings.EMBEDDING_DIMENSION:
            raise ValueError(
                f"Dimension mismatch at embedding {index}: model returned {len(embedding)}D "
                f"but EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION}"
            )


def _delete_existing_chunks(session, article_id: int) -> None:
    session.query(ArticleChunk).filter(ArticleChunk.article_id == article_id).delete(
        synchronize_session=False
    )
    session.flush()


def _heading_path_to_text(heading_path: Iterable[str] | None) -> str:
    if not heading_path:
        return ""
    return json.dumps(list(heading_path), ensure_ascii=False)


def _heading_path_from_text(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [str(item) for item in loaded]
    except Exception:
        pass
    return [part.strip() for part in raw.split(">") if part.strip()]


def _safe_section(section: str | None) -> str:
    normalized = normalize_section_name(section)
    if len(normalized) > 128:
        raise ValueError(f"Normalized section exceeds 128 characters: {normalized!r}")
    return normalized


def ingest_article_chunks(article_id: int, chunks: list[str]) -> dict:
    """
    Legacy fallback ingestion for plain string chunks.
    """
    if not chunks:
        return {"stored": False, "chunk_count": 0, "message": "No chunks provided"}

    paper_chunks = [
        PaperChunk(
            content=chunk_text,
            content_for_embedding=chunk_text,
            chunk_index=i,
            chunk_type="section_text",
            section="unknown",
            section_title="Unknown",
            section_level=1,
            heading_path=["Unknown"],
            token_count=len(chunk_text.split()),
            metadata={"legacy": True},
            chunking_version="v1",
        )
        for i, chunk_text in enumerate(chunks)
    ]
    return ingest_paper_chunks(article_id, paper_chunks)


def ingest_paper_chunks(article_id: int, chunks: list[PaperChunk]) -> dict:
    """
    Store structured paper chunks and Google embeddings in database.
    """
    if not chunks:
        return {"stored": False, "chunk_count": 0, "message": "No chunks provided"}

    ensure_pgvector_schema()

    with SessionLocal() as session:
        try:
            _delete_existing_chunks(session, article_id)

            embedding_inputs = [chunk.content_for_embedding or chunk.content for chunk in chunks]
            embeddings_list = embed_texts(embedding_inputs)
            if len(embeddings_list) != len(chunks):
                raise ValueError(
                    f"Embedding count mismatch: received {len(embeddings_list)} vectors "
                    f"for {len(chunks)} chunks"
                )
            _validate_embedding_dimensions(embeddings_list)

            type_counts = Counter(chunk.chunk_type for chunk in chunks)
            for i, (chunk, embedding_vector) in enumerate(zip(chunks, embeddings_list)):
                db_chunk = ArticleChunk(
                    article_id=article_id,
                    chunk_index=i,
                    content=chunk.content,
                    section=_safe_section(chunk.section or chunk.section_title),
                    section_title=chunk.section_title,
                    chunk_type=chunk.chunk_type,
                    heading_path=_heading_path_to_text(chunk.heading_path),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    token_count=chunk.token_count,
                    metadata_json=chunk.metadata,
                    chunking_version=chunk.chunking_version or settings.CHUNKING_VERSION,
                )
                session.add(db_chunk)
                session.flush()

                db_embedding = Embedding(chunk_id=db_chunk.id, embedding=embedding_vector)
                session.add(db_embedding)

            session.commit()
            return {
                "stored": True,
                "chunk_count": len(chunks),
                "chunking_version": settings.CHUNKING_VERSION,
                "embedding_provider": settings.EMBEDDING_PROVIDER,
                "embedding_model": settings.GOOGLE_EMBEDDING_MODEL,
                "chunk_type_counts": dict(type_counts),
            }
        except Exception as e:
            session.rollback()
            logger.error(f"ingest_paper_chunks failed: {e}\n{traceback.format_exc()}")
            return {"stored": False, "chunk_count": 0, "error": str(e)}


def _row_to_result(row) -> dict:
    chunk = row.ArticleChunk
    return {
        "chunk_id": chunk.id,
        "content": chunk.content,
        "distance": row.distance,
        "section": chunk.section,
        "section_title": chunk.section_title,
        "chunk_type": chunk.chunk_type,
        "heading_path": _heading_path_from_text(chunk.heading_path),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
        "metadata": chunk.metadata_json or {},
    }


def _run_similarity_query(session, article_id: int, query_embedding: list[float], limit: int, sections=None):
    query = (
        session.query(ArticleChunk, Embedding.embedding.l2_distance(query_embedding).label("distance"))
        .join(Embedding)
        .filter(ArticleChunk.article_id == article_id)
    )
    if sections:
        query = query.filter(ArticleChunk.section.in_(sections))
    return query.order_by("distance").limit(limit).all()


def similarity_search(
    article_id: int,
    query_embedding: list[float],
    limit: int = 5,
    focus_sections: Optional[list[str]] = None,
    min_results: int = 3,
):
    """
    Query for similar chunks using vector similarity, optionally preferring paper sections.
    """
    normalized_sections = [normalize_section_name(section) for section in (focus_sections or []) if section]
    normalized_sections = [section for section in normalized_sections if section and section != "unknown"]

    with SessionLocal() as session:
        if normalized_sections:
            focused_rows = _run_similarity_query(
                session,
                article_id=article_id,
                query_embedding=query_embedding,
                limit=limit,
                sections=normalized_sections,
            )
            if len(focused_rows) >= min_results:
                return [_row_to_result(row) for row in focused_rows]

            fallback_rows = _run_similarity_query(
                session,
                article_id=article_id,
                query_embedding=query_embedding,
                limit=limit,
                sections=None,
            )
            seen = set()
            merged = []
            for row in focused_rows + fallback_rows:
                chunk_id = row.ArticleChunk.id
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                merged.append(row)
                if len(merged) >= limit:
                    break
            return [_row_to_result(row) for row in merged]

        rows = _run_similarity_query(
            session,
            article_id=article_id,
            query_embedding=query_embedding,
            limit=limit,
            sections=None,
        )
        return [_row_to_result(row) for row in rows]
