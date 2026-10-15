import json
import traceback
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable, Optional

from sqlalchemy import and_, column, inspect, or_, table, text
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.core.logger import get_logger
from app.embeddings.embedder import embed_texts
from app.embeddings.models import ArticleChunk, Base, Embedding
from app.pdf_pipeline.chunker import PaperChunk, normalize_section_name

logger = get_logger(__name__)
_articles = table(
    "articles",
    column("id"),
    column("pdf_generation"),
    column("content_revision"),
)
_PDF_GENERATION_METADATA_KEY = "__deepscholar_pdf_generation"
_PDF_CONTENT_REVISION_METADATA_KEY = "__deepscholar_content_revision"

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


def ingest_article_chunks(
    article_id: int,
    chunks: list[str],
    *,
    pdf_generation: int | None = None,
    content_revision: int | None = None,
) -> dict:
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
    return ingest_paper_chunks(
        article_id,
        paper_chunks,
        pdf_generation=pdf_generation,
        content_revision=content_revision,
    )


def _is_retryable_ingest_error(error: Exception) -> bool:
    if isinstance(error, (OperationalError, InterfaceError)):
        return True
    if isinstance(error, DBAPIError):
        return bool(error.connection_invalidated)
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    message = str(error).upper()
    return code in {429, 500, 502, 503, 504} or any(
        marker in message
        for marker in ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED", "TIMEOUT")
    )


def ingest_paper_chunks(
    article_id: int,
    chunks: list[PaperChunk],
    *,
    pdf_generation: int | None = None,
    content_revision: int | None = None,
) -> dict:
    """
    Store structured paper chunks and Google embeddings in database.
    """
    if not chunks:
        return {"stored": False, "chunk_count": 0, "message": "No chunks provided"}

    try:
        ensure_pgvector_schema()
        embedding_inputs = [chunk.content_for_embedding or chunk.content for chunk in chunks]
        embeddings_list = embed_texts(embedding_inputs)
        if len(embeddings_list) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: received {len(embeddings_list)} vectors "
                f"for {len(chunks)} chunks"
            )
        _validate_embedding_dimensions(embeddings_list)

        with SessionLocal() as session:
            if pdf_generation is not None:
                lock_suffix = " FOR UPDATE" if session.bind.dialect.name == "postgresql" else ""
                article_row = session.execute(
                    text(
                        "SELECT pdf_generation, content_revision, pdf_generation_deadline FROM articles "
                        "WHERE id = :article_id" + lock_suffix
                    ),
                    {"article_id": article_id},
                ).first()
                if not article_row:
                    return {
                        "stored": False,
                        "chunk_count": 0,
                        "error_code": "article_not_found",
                        "error": "Article not found for PDF embedding write.",
                        "retryable": False,
                    }
                if (
                    article_row.pdf_generation != pdf_generation
                    or article_row.content_revision != content_revision
                ):
                    return {
                        "stored": False,
                        "chunk_count": 0,
                        "error_code": "stale_pdf_generation",
                        "error": "PDF embedding write belongs to a stale article revision.",
                        "retryable": False,
                    }
                deadline_at = article_row.pdf_generation_deadline
                if deadline_at is None:
                    return {
                        "stored": False,
                        "chunk_count": 0,
                        "error_code": "pdf_generation_expired",
                        "error": "The PDF generation has no active deadline.",
                        "retryable": False,
                    }
                if deadline_at.tzinfo is None:
                    deadline_at = deadline_at.replace(tzinfo=timezone.utc)
                if deadline_at <= datetime.now(timezone.utc):
                    return {
                        "stored": False,
                        "chunk_count": 0,
                        "error_code": "pdf_generation_expired",
                        "error": "The PDF generation deadline has expired.",
                        "retryable": False,
                    }

            _delete_existing_chunks(session, article_id)
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
                    metadata_json=(
                        {
                            **(chunk.metadata or {}),
                            _PDF_GENERATION_METADATA_KEY: pdf_generation,
                            _PDF_CONTENT_REVISION_METADATA_KEY: content_revision,
                        }
                        if pdf_generation is not None and content_revision is not None
                        else chunk.metadata
                    ),
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
        logger.error(f"ingest_paper_chunks failed: {e}\n{traceback.format_exc()}")
        return {
            "stored": False,
            "chunk_count": 0,
            "error": str(e),
            "retryable": _is_retryable_ingest_error(e),
        }


def _row_to_result(row) -> dict:
    chunk = row.ArticleChunk
    metadata = dict(chunk.metadata_json or {})
    metadata.pop(_PDF_GENERATION_METADATA_KEY, None)
    metadata.pop(_PDF_CONTENT_REVISION_METADATA_KEY, None)
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
        "metadata": metadata,
    }


def _current_pdf_chunk_filter():
    stored_generation = ArticleChunk.metadata_json[_PDF_GENERATION_METADATA_KEY].as_integer()
    stored_revision = ArticleChunk.metadata_json[_PDF_CONTENT_REVISION_METADATA_KEY].as_integer()
    current_pdf_chunks = and_(
        stored_generation == _articles.c.pdf_generation,
        stored_revision == _articles.c.content_revision,
    )
    legacy_chunks = and_(
        _articles.c.pdf_generation == 0,
        _articles.c.content_revision == 0,
        stored_generation.is_(None),
        stored_revision.is_(None),
    )
    return or_(current_pdf_chunks, legacy_chunks)


def _run_similarity_query(session, article_id: int, query_embedding: list[float], limit: int, sections=None):
    query = (
        session.query(ArticleChunk, Embedding.embedding.l2_distance(query_embedding).label("distance"))
        .join(Embedding)
        .join(_articles, ArticleChunk.article_id == _articles.c.id)
        .filter(ArticleChunk.article_id == article_id)
        .filter(_current_pdf_chunk_filter())
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
