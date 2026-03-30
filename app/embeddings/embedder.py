import os
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Returns the configured Langchain Embeddings model based on the chosen provider.
    """
    provider = (settings.EMBEDDING_PROVIDER or "google").lower()

    if provider == "huggingface":
        model_name = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        print(f"[Embedder] Using HuggingFace model: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "google":
        # models/gemini-embedding-001 is available in the user's current API key list.
        # models/text-embedding-004 was confirmed NOT-FOUND (404).
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        print(f"[Embedder] Using Google model: {model_name} (Stable v2.x SDK)")
        
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            task_type="retrieval_document",
        )
    else:
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.OPENAI_API_KEY,
        )

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using the selected provider.
    """
    if not texts:
        return []
    embeddings_model = get_embeddings()
    return embeddings_model.embed_documents(texts)
