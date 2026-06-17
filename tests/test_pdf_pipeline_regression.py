import os
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("INTERNAL_SERVICE_KEY", "test-key")

from fastapi import BackgroundTasks

from app.api import pdf
from app.pdf_pipeline.chunker import PaperChunk


def _request():
    return pdf.PDFUploadRequest(
        pdf_url="https://example.com/paper.pdf",
        slug="sample-paper",
        article_id=42,
    )


def _paper_chunk():
    return PaperChunk(
        content="chunk",
        content_for_embedding="Title: T\n\nchunk",
        chunk_index=0,
        chunk_type="abstract",
        section="abstract",
        section_title="Abstract",
        section_level=1,
        heading_path=["T", "Abstract"],
        token_count=5,
        metadata={},
        chunking_version="v2",
    )


def _patch_success_common(monkeypatch):
    monkeypatch.setattr(pdf, "_download_pdf_from_r2", lambda _url: b"%PDF fake")
    monkeypatch.setattr(
        pdf,
        "extract_sections_with_llamaparse",
        lambda _bytes: {
            "title": "Extracted Title",
            "abstract": "Extracted abstract.",
            "content": "## Introduction\nBody with enough extracted scientific paper content to pass the pipeline validation.",
        },
    )
    patch_response = MagicMock(ok=True, status_code=200, text="ok")
    patch_mock = MagicMock(return_value=patch_response)
    monkeypatch.setattr(pdf.requests, "patch", patch_mock)
    return patch_mock


def test_sync_upload_response_contract(monkeypatch):
    monkeypatch.setattr(
        pdf,
        "process_pdf_pipeline",
        lambda _request: {"status": "success", "title": "T", "abstract": "A", "content": "C"},
    )

    response = pdf.upload_pdf(_request(), BackgroundTasks(), sync=True)

    assert response["status"] == "completed"
    assert response["slug"] == "sample-paper"
    assert response["data"]["status"] == "success"


def test_async_upload_response_contract():
    response = pdf.upload_pdf(_request(), BackgroundTasks(), sync=False)

    assert response["status"] == "processing_started"
    assert response["slug"] == "sample-paper"
    assert "queued" in response["message"]


def test_pipeline_uses_structured_chunking_and_keeps_backend_patch_payload(monkeypatch):
    patch_mock = _patch_success_common(monkeypatch)
    chunk_mock = MagicMock(return_value=[_paper_chunk()])
    ingest_mock = MagicMock(return_value={"stored": True, "chunk_count": 1})
    monkeypatch.setattr(pdf, "chunk_paper", chunk_mock)
    monkeypatch.setattr(pdf, "ingest_paper_chunks", ingest_mock)

    result = pdf.process_pdf_pipeline(_request())

    assert result["status"] == "success"
    chunk_mock.assert_called_once()
    ingest_mock.assert_called_once()
    payload = patch_mock.call_args.kwargs["json"]
    assert payload == {
        "title": "Extracted Title",
        "abstract": "Extracted abstract.",
        "content": "## Introduction\nBody with enough extracted scientific paper content to pass the pipeline validation.",
        "pdf_url": "https://example.com/paper.pdf",
    }


def test_pipeline_falls_back_to_legacy_chunker_when_structured_chunking_fails(monkeypatch):
    _patch_success_common(monkeypatch)
    monkeypatch.setattr(pdf, "chunk_paper", MagicMock(side_effect=RuntimeError("chunker failed")))
    legacy_chunk_mock = MagicMock(return_value=["legacy chunk"])
    legacy_ingest_mock = MagicMock(return_value={"stored": True, "chunk_count": 1})
    monkeypatch.setattr(pdf, "chunk_text", legacy_chunk_mock)
    monkeypatch.setattr(pdf, "ingest_article_chunks", legacy_ingest_mock)

    result = pdf.process_pdf_pipeline(_request())

    assert result["status"] == "success"
    legacy_chunk_mock.assert_called_once()
    legacy_ingest_mock.assert_called_once_with(42, ["legacy chunk"])


def test_pipeline_uses_pypdf2_llm_fallback_when_llamaparse_insufficient(monkeypatch):
    monkeypatch.setattr(pdf, "_download_pdf_from_r2", lambda _url: b"%PDF fake")
    monkeypatch.setattr(pdf, "extract_sections_with_llamaparse", lambda _bytes: None)
    monkeypatch.setattr(pdf, "extract_text_from_pdf", lambda _bytes: "Raw content\nReferences\nnoisy ref")
    monkeypatch.setattr(pdf, "extract_metadata_from_text", lambda _text: {"title": "Fallback T", "abstract": "Fallback A"})
    monkeypatch.setattr(pdf, "remove_references", lambda text: text.split("References")[0].strip())
    monkeypatch.setattr(pdf.requests, "patch", MagicMock(return_value=MagicMock(ok=True, status_code=200, text="ok")))
    monkeypatch.setattr(pdf, "chunk_paper", MagicMock(return_value=[_paper_chunk()]))
    monkeypatch.setattr(pdf, "ingest_paper_chunks", MagicMock(return_value={"stored": True, "chunk_count": 1}))

    result = pdf.process_pdf_pipeline(_request())

    assert result["status"] == "success"
    assert result["title"] == "Fallback T"
    assert result["content"] == "Raw content"


def test_pipeline_returns_partial_success_when_embedding_ingest_fails(monkeypatch):
    _patch_success_common(monkeypatch)
    monkeypatch.setattr(pdf, "chunk_paper", MagicMock(return_value=[_paper_chunk()]))
    monkeypatch.setattr(pdf, "ingest_paper_chunks", MagicMock(return_value={"stored": False, "error": "embedding failed"}))

    result = pdf.process_pdf_pipeline(_request())

    assert result["status"] == "partial_success"
    assert result["embedding_error"] == "embedding failed"
