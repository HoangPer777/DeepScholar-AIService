from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.core.config import settings
from app.core.database import Base


class ArticleChunk(Base):
    __tablename__ = "article_chunks"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    section = Column(String(128), index=True)
    section_title = Column(Text)
    chunk_type = Column(String(64), index=True)
    heading_path = Column(Text)
    page_start = Column(Integer)
    page_end = Column(Integer)
    token_count = Column(Integer)
    metadata_json = Column("metadata", JSON)
    chunking_version = Column(String(32), nullable=False, default="v2")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_article_chunks_article_section", "article_id", "section"),
        Index("idx_article_chunks_article_type", "article_id", "chunk_type"),
    )


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("article_chunks.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
