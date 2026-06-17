import os
import random
import time

from app.core.config import settings


_GENAI_SDK_MODELS = {"models/gemini-embedding-2-preview", "gemini-embedding-2-preview"}


def get_embeddings():
    """
    Return the configured embedding implementation.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "google":
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", settings.GOOGLE_EMBEDDING_MODEL)
        if model_name in _GENAI_SDK_MODELS or model_name.replace("models/", "") in _GENAI_SDK_MODELS:
            print(f"[Embedder] Using google-genai SDK for model: {model_name}")
            return _GoogleGenAIEmbeddings(model=model_name, api_key=settings.GOOGLE_API_KEY)

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        print(f"[Embedder] Using langchain-google-genai for model: {model_name}")
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = settings.EMBED_MODEL
        print(f"[Embedder] Using HuggingFace model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)

    from langchain_openai import OpenAIEmbeddings

    model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    print(f"[Embedder] Using OpenAI model: {model_name}")
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=settings.OPENAI_API_KEY,
    )


class _GoogleGenAIEmbeddings:
    """
    google-genai wrapper with request pacing and bounded retry.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.api_key = api_key

    def _get_client(self):
        from google import genai

        return genai.Client(api_key=self.api_key)

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
        message = str(error).upper()
        return status_code in {429, 500, 502, 503, 504} or any(
            marker in message
            for marker in ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")
        )

    def _embed_with_retry(self, client, contents, task_type: str):
        from google.genai import types

        max_retries = max(0, settings.GOOGLE_EMBEDDING_MAX_RETRIES)
        for attempt in range(max_retries + 1):
            try:
                return client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            except Exception as error:
                if not self._is_retryable_error(error) or attempt >= max_retries:
                    raise
                base_delay = settings.GOOGLE_EMBEDDING_RETRY_BASE_SECONDS * (2**attempt)
                delay = min(base_delay, 60.0) + random.uniform(0, 1)
                print(
                    f"[Embedder] Retryable error ({attempt + 1}/{max_retries}); "
                    f"retrying in {delay:.1f}s: {error}"
                )
                time.sleep(delay)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        result = []
        total = len(texts)

        for index, text in enumerate(texts, start=1):
            response = self._embed_with_retry(
                client,
                contents=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
            if len(response.embeddings) != 1:
                raise ValueError(
                    f"Google embedding request returned {len(response.embeddings)} vectors "
                    "for one text"
                )
            result.append(list(response.embeddings[0].values))
            if index == 1 or index % 10 == 0 or index == total:
                print(f"[Embedder] Embedded document {index}/{total}")
            if index < total and settings.GOOGLE_EMBEDDING_REQUEST_DELAY_SECONDS > 0:
                time.sleep(settings.GOOGLE_EMBEDDING_REQUEST_DELAY_SECONDS)

        return result

    def embed_query(self, text: str) -> list[float]:
        client = self._get_client()
        response = self._embed_with_retry(
            client,
            contents=[text],
            task_type="RETRIEVAL_QUERY",
        )
        return list(response.embeddings[0].values)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using the selected provider.
    """
    if not texts:
        return []
    return get_embeddings().embed_documents(texts)
