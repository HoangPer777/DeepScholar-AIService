import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("INTERNAL_SERVICE_KEY", "test-key")
os.environ["EMBEDDING_PROVIDER"] = "google"
os.environ["GOOGLE_EMBEDDING_MODEL"] = "models/gemini-embedding-2-preview"
os.environ["EMBEDDING_DIMENSION"] = "3072"


def test_google_embedding_provider_and_model_are_configured():
    from app.core.config import settings

    assert settings.EMBEDDING_PROVIDER == "google"
    assert os.getenv("GOOGLE_EMBEDDING_MODEL") == "models/gemini-embedding-2-preview"
    assert settings.EMBEDDING_DIMENSION == 3072


def test_gemini_embedding_2_preview_uses_google_genai_wrapper():
    from app.embeddings import embedder

    embeddings = embedder.get_embeddings()

    assert embeddings.__class__.__name__ == "_GoogleGenAIEmbeddings"
    assert embeddings.model == "models/gemini-embedding-2-preview"


def test_embedding_dimension_validation_accepts_3072():
    from app.embeddings.vector_store import _validate_embedding_dimensions

    _validate_embedding_dimensions([[0.0] * 3072])


def test_embedding_dimension_validation_rejects_wrong_dimension():
    from app.embeddings.vector_store import _validate_embedding_dimensions

    try:
        _validate_embedding_dimensions([[0.0] * 128])
    except ValueError as exc:
        assert "Dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected dimension mismatch ValueError")
