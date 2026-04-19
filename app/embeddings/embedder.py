import os
from app.core.config import settings

# Models that require the google-genai SDK (not supported by langchain-google-genai)
_GENAI_SDK_MODELS = {"models/gemini-embedding-2-preview", "gemini-embedding-2-preview"}


def get_embeddings():
    """
    Returns the configured embeddings model based on EMBEDDING_PROVIDER.
    Supported: "openai", "google", "huggingface"

    For Google provider:
    - gemini-embedding-2-preview → uses google-genai SDK directly
    - other models (gemini-embedding-001, etc.) → uses langchain-google-genai
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "google":
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
        if model_name in _GENAI_SDK_MODELS or model_name.replace("models/", "") in _GENAI_SDK_MODELS:
            print(f"[Embedder] Using google-genai SDK for model: {model_name}")
            return _GoogleGenAIEmbeddings(model=model_name, api_key=settings.GOOGLE_API_KEY)
        else:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            print(f"[Embedder] Using langchain-google-genai for model: {model_name}")
            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=settings.GOOGLE_API_KEY,
            )

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = settings.EMBED_MODEL
        print(f"[Embedder] Using HuggingFace model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)

    else:
        # Default: OpenAI
        from langchain_openai import OpenAIEmbeddings
        model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        print(f"[Embedder] Using OpenAI model: {model_name}")
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.OPENAI_API_KEY,
        )


class _GoogleGenAIEmbeddings:
    """
    Wrapper around google-genai SDK for models like gemini-embedding-2-preview.
    Uses the Gemini API v1 directly.
    """

    def __init__(self, model: str, api_key: str):
        # Normalize model name — API accepts both with and without "models/" prefix
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.api_key = api_key

    def _get_client(self):
        from google import genai
        return genai.Client(api_key=self.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types
        client = self._get_client()
        result = []
        # Batch in groups of 100 (API limit)
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            response = client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            for emb in response.embeddings:
                result.append(list(emb.values))
        return result

    def embed_query(self, text: str) -> list[float]:
        from google.genai import types
        client = self._get_client()
        response = client.models.embed_content(
            model=self.model,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )
        return list(response.embeddings[0].values)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using the selected provider.
    """
    if not texts:
        return []
    return get_embeddings().embed_documents(texts)
