from types import SimpleNamespace


class RetryableError(Exception):
    status_code = 429


def _response(count: int):
    return SimpleNamespace(
        embeddings=[SimpleNamespace(values=[float(index)]) for index in range(count)]
    )


def test_google_embedder_embeds_documents_individually(monkeypatch):
    from app.embeddings import embedder

    calls = []

    class Models:
        def embed_content(self, **kwargs):
            calls.append(kwargs["contents"])
            return _response(len(kwargs["contents"]))

    wrapper = embedder._GoogleGenAIEmbeddings("gemini-embedding-2-preview", "key")
    monkeypatch.setattr(wrapper, "_get_client", lambda: SimpleNamespace(models=Models()))
    monkeypatch.setattr(embedder.settings, "GOOGLE_EMBEDDING_REQUEST_DELAY_SECONDS", 0)

    vectors = wrapper.embed_documents(["a", "b", "c"])

    assert calls == ["a", "b", "c"]
    assert len(vectors) == 3


def test_google_embedder_retries_429(monkeypatch):
    from app.embeddings import embedder

    attempts = 0

    class Models:
        def embed_content(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RetryableError("429 RESOURCE_EXHAUSTED")
            return _response(len(kwargs["contents"]))

    wrapper = embedder._GoogleGenAIEmbeddings("gemini-embedding-2-preview", "key")
    monkeypatch.setattr(wrapper, "_get_client", lambda: SimpleNamespace(models=Models()))
    monkeypatch.setattr(embedder.settings, "GOOGLE_EMBEDDING_MAX_RETRIES", 3)
    monkeypatch.setattr(embedder.settings, "GOOGLE_EMBEDDING_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(embedder.settings, "GOOGLE_EMBEDDING_REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(embedder.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(embedder.random, "uniform", lambda _start, _end: 0)

    vectors = wrapper.embed_documents(["a"])

    assert attempts == 3
    assert len(vectors) == 1


def test_google_embedder_stops_after_retry_limit(monkeypatch):
    from app.embeddings import embedder

    attempts = 0

    class Models:
        def embed_content(self, **kwargs):
            nonlocal attempts
            attempts += 1
            raise RetryableError("429 RESOURCE_EXHAUSTED")

    wrapper = embedder._GoogleGenAIEmbeddings("gemini-embedding-2-preview", "key")
    monkeypatch.setattr(wrapper, "_get_client", lambda: SimpleNamespace(models=Models()))
    monkeypatch.setattr(embedder.settings, "GOOGLE_EMBEDDING_MAX_RETRIES", 2)
    monkeypatch.setattr(embedder.settings, "GOOGLE_EMBEDDING_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(embedder.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(embedder.random, "uniform", lambda _start, _end: 0)

    try:
        wrapper.embed_documents(["a"])
    except RetryableError:
        pass
    else:
        raise AssertionError("Expected retryable error after retry limit")

    assert attempts == 3
