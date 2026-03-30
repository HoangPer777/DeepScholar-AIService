import os
from typing import Dict, List

from app.embeddings.embedder import embed_texts
from app.embeddings.models import ArticleChunk, Embedding
from app.embeddings.vector_store import SessionLocal


def vector_search(query: str, top_k: int = 5) -> List[Dict]:
    use_mock = os.getenv("USE_MOCK_VECTOR", "false").lower() == "true"
    if use_mock:
        return _mock_vector_search(query, top_k)
    try:
        return _pgvector_search(query, top_k)
    except Exception as exc:
        raise RuntimeError(
            "Vector search failed. Check PGVector connection/dependencies and embedding model config. "
            f"Original error: {exc}"
        ) from exc


def ingest_documents(documents: List[Dict]) -> int:
    from langchain_core.documents import Document

    store = _get_vectorstore()
    docs = [
        Document(page_content=d.get("text", ""), metadata={k: v for k, v in d.items() if k != "text"})
        for d in documents
    ]
    store.add_documents(docs)
    return len(docs)


def _get_vectorstore():
    from langchain_postgres import PGVector
    from langchain_postgres.vectorstores import DistanceStrategy

    return PGVector(
        embeddings=_get_embeddings(),
        collection_name=os.getenv("PG_COLLECTION", "deepscholar_documents"),
        connection=_get_connection_string(),
        distance_strategy=DistanceStrategy.COSINE,
        pre_delete_collection=False,
    )


def _pgvector_search(query: str, top_k: int) -> List[Dict]:
    # Prefer LangChain PGVector if available; fallback to direct SQLAlchemy pgvector query.
    try:
        store = _get_vectorstore()
        results = store.similarity_search_with_relevance_scores(query=query, k=top_k)
        output: List[Dict] = []
        for doc, score in results:
            meta = doc.metadata or {}
            output.append(
                {
                    "text": doc.page_content,
                    "source": meta.get("source", "pgvector"),
                    "title": meta.get("title", ""),
                    "page": meta.get("page", ""),
                    "authors": meta.get("authors", ""),
                    "doi": meta.get("doi", ""),
                    "url": meta.get("url", ""),
                    "score": round(float(score), 4),
                }
            )
        return output
    except Exception:
        return _sqlalchemy_pgvector_search(query=query, top_k=top_k)


def _sqlalchemy_pgvector_search(query: str, top_k: int) -> List[Dict]:
    query_embeddings = embed_texts([query])
    if not query_embeddings:
        return []
    query_embedding = query_embeddings[0]

    with SessionLocal() as session:
        rows = (
            session.query(ArticleChunk, Embedding.embedding.cosine_distance(query_embedding).label("distance"))
            .join(Embedding, Embedding.chunk_id == ArticleChunk.id)
            .order_by("distance")
            .limit(top_k)
            .all()
        )

    output: List[Dict] = []
    for row in rows:
        chunk = row.ArticleChunk
        distance = float(row.distance)
        score = max(0.0, min(1.0, 1.0 - distance))
        output.append(
            {
                "text": chunk.content,
                "source": "pgvector_sqlalchemy",
                "title": f"Article {chunk.article_id} chunk {chunk.chunk_index}",
                "page": "",
                "authors": "",
                "doi": "",
                "url": "",
                "score": round(score, 4),
            }
        )
    return output


def _get_embeddings():
    embed_provider = (
        os.getenv("EMBED_PROVIDER")
        or os.getenv("EMBEDDING_PROVIDER")
        or "huggingface"
    ).lower()
    if embed_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if embed_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    raise ValueError("Unsupported EMBED_PROVIDER")


def _get_connection_string() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg://", 1)
        if url.startswith("postgresql://") and "+psycopg" not in url:
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "deepscholar")
    password = os.getenv("POSTGRES_PASSWORD", "deepscholar")
    db = os.getenv("POSTGRES_DB", "deepscholar")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


def _mock_vector_search(query: str, top_k: int) -> List[Dict]:
    return [
        {
            "text": f"[MOCK] PGVector result for '{query}'",
            "source": "mock_pgvector",
            "title": f"Mock Paper: {query}",
            "authors": "Nguyen et al., 2024",
            "doi": "10.xxxx/mock",
            "score": 0.92,
        },
        {
            "text": f"[MOCK] Additional PGVector context for '{query}'",
            "source": "mock_pgvector",
            "title": f"Mock Survey: {query}",
            "authors": "Tran et al., 2024",
            "doi": "10.xxxx/mock2",
            "score": 0.85,
        },
    ][:top_k]