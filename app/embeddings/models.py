from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.sql import func
from app.core.database import Base
from pgvector.sqlalchemy import Vector
from app.core.config import settings

class ArticleChunk(Base):
    __tablename__ = "article_chunks"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, index=True) # Maps to the Article ID in the Django backend
    chunk_index = Column(Integer)
    content = Column(Text)

class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("article_chunks.id", ondelete="CASCADE"))
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
