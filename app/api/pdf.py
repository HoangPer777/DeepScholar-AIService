import hmac
import os
import time
import urllib.parse
import uuid
from typing import Callable, Optional

import boto3
import requests
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.job_store import get_job_store
from app.core.logger import get_logger
from app.pdf_pipeline.llama_extractor import extract_sections_with_llamaparse, remove_references
from app.pdf_pipeline.extractor import extract_text_from_pdf
from app.pdf_pipeline.llm_extractor import extract_metadata_from_text
from app.pdf_pipeline.chunker import chunk_paper, chunk_text
from app.embeddings.vector_store import ingest_article_chunks, ingest_paper_chunks

router = APIRouter()
logger = get_logger(__name__)
_job_store = get_job_store()
MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024

ProgressCallback = Callable[[dict], None]


def _emit_progress(callback: Optional[ProgressCallback], phase: str, message: str, percent: int) -> None:
    if callback is None:
        return
    try:
        callback({"phase": phase, "message": message, "percent": percent})
    except Exception:
        logger.warning("PDF progress update failed for phase %s", phase)


def _job_progress(phase: str, message: str, percent: int) -> dict:
    return {"phase": phase, "message": message, "percent": percent}


def _require_internal_service_key(provided_key: Optional[str]) -> None:
    expected_key = settings.INTERNAL_SERVICE_KEY
    if not expected_key or not provided_key or not hmac.compare_digest(provided_key, expected_key):
        raise HTTPException(status_code=403, detail="Internal service authorization required")


class PDFUploadRequest(BaseModel):
    task_id: str
    generation: int
    content_revision: int
    idempotency_key: str
    pdf_url: str
    slug: str
    article_id: int


def _create_pdf_job(task_id: str, request: PDFUploadRequest) -> bool:
    initial_data = {
        "kind": "pdf",
        "task_id": task_id,
        "status": "queued",
        "slug": request.slug,
        "article_id": request.article_id,
        "generation": request.generation,
        "content_revision": request.content_revision,
        "idempotency_key": request.idempotency_key,
        "progress": _job_progress("queued", "PDF processing queued.", 0),
    }
    created = _job_store.create_job_if_absent(task_id, initial_data)
    if not created:
        existing = _job_store.get_pdf_job(task_id)
        same_identity = bool(existing) and all(
            existing.get(name) == initial_data[name]
            for name in ("article_id", "generation", "content_revision", "idempotency_key", "slug")
        )
        if not same_identity:
            raise HTTPException(status_code=409, detail="PDF task identity conflict")
    return created


def _update_pdf_job(task_id: str, **updates) -> None:
    _job_store.update_pdf_job(task_id, updates)


def _run_pdf_job(task_id: str, request: PDFUploadRequest) -> None:
    _update_pdf_job(
        task_id,
        status="running",
        progress=_job_progress("starting", "Starting PDF processing…", 5),
    )

    def on_progress(progress: dict) -> None:
        _update_pdf_job(task_id, status="running", progress=progress)

    try:
        result = process_pdf_pipeline(request, progress_callback=on_progress)
        if result.get("status") == "success":
            _update_pdf_job(
                task_id,
                status="completed",
                progress=_job_progress("completed", "PDF processing completed.", 100),
            )
            return
        _update_pdf_job(
            task_id,
            status="failed",
            progress=_job_progress("failed", "PDF processing failed.", 100),
            error_code=result.get("error_code", "pdf_processing_failed"),
            message=result.get("error", "PDF processing failed."),
            retryable=bool(result.get("retryable", False)),
            article_saved=bool(result.get("article_saved", False)),
        )
    except Exception:
        logger.error("Unhandled PDF job failure for task %s", task_id)
        _update_pdf_job(
            task_id,
            status="failed",
            progress=_job_progress("failed", "PDF processing failed.", 100),
            error_code="pdf_processing_failed",
            message="PDF processing failed. Please try again.",
            retryable=False,
        )


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_S3_REGION_NAME", "apac")
    )


def _download_pdf_from_r2(pdf_url: str) -> bytes:
    """
    Download a PDF file from Cloudflare R2 using boto3.
    Supports both public URL format and private endpoint format.
    """
    parsed_url = urllib.parse.urlparse(pdf_url)
    path_parts = parsed_url.path.lstrip('/').split('/', 1)

    bucket_name = os.getenv("AWS_STORAGE_BUCKET_NAME", "deepscholar-articles")

    if len(path_parts) == 2:
        # URL format: https://<endpoint>/<bucket>/<key>  OR  https://<pub>.r2.dev/<key>
        possible_bucket = path_parts[0]
        possible_key = urllib.parse.unquote(path_parts[1])

        # If the first path segment is the bucket name, use it
        if possible_bucket == bucket_name:
            object_key = possible_key
        else:
            # Public R2 URL: https://pub-xxx.r2.dev/articles/uuid_file.pdf
            object_key = urllib.parse.unquote(parsed_url.path.lstrip('/'))
    else:
        object_key = urllib.parse.unquote(parsed_url.path.lstrip('/'))

    s3 = get_s3_client()
    print(f"Downloading s3://{bucket_name}/{object_key}")

    try:
        metadata = s3.head_object(Bucket=bucket_name, Key=object_key)
        content_length = metadata.get('ContentLength')
        try:
            content_length = int(content_length) if content_length is not None else None
        except (TypeError, ValueError):
            content_length = None
        if content_length is not None and content_length > MAX_PDF_SIZE_BYTES:
            _delete_oversized_object(s3, bucket_name, object_key)
            raise ValueError("PDF file exceeds the 25 MiB limit")
    except ValueError:
        raise
    except Exception:
        # A missing/unavailable metadata response is not trusted as proof of safety.
        # The bounded streaming read below remains mandatory.
        logger.warning("PDF size metadata unavailable; enforcing streaming cap")

    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    body = response['Body']
    chunks = []
    total_bytes = 0
    try:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_PDF_SIZE_BYTES:
                _delete_oversized_object(s3, bucket_name, object_key)
                raise ValueError("PDF file exceeds the 25 MiB limit")
            chunks.append(chunk)
    finally:
        close = getattr(body, 'close', None)
        if close:
            close()

    file_bytes = b''.join(chunks)
    print(f"Downloaded {len(file_bytes)} bytes from Cloudflare R2.")
    return file_bytes


def _delete_oversized_object(s3, bucket_name: str, object_key: str) -> None:
    try:
        s3.delete_object(Bucket=bucket_name, Key=object_key)
        logger.warning("Deleted oversized PDF object %s", object_key)
    except Exception:
        logger.error("Unable to delete oversized PDF object")


def process_pdf_pipeline(
    request: PDFUploadRequest,
    progress_callback: Optional[ProgressCallback] = None,
):
    """
    Full AI extraction pipeline.

    Flow:
    1. Download PDF from Cloudflare R2 via boto3
    2. Extract title/abstract/content via LlamaParse (IEEE-aware)
       → Falls back to PyPDF2 + LLM if LlamaParse is unavailable
    3. Persist extracted fields through the idempotent PDF callback
    4. Chunk content and store in PGVector
    """
    print(f"[Pipeline] Starting for article: {request.slug}")
    article_saved = False
    try:
        # ── Step 1: Download PDF ──────────────────────────────────────────────
        _emit_progress(progress_callback, "downloading", "Downloading PDF…", 15)
        file_bytes = _download_pdf_from_r2(request.pdf_url)

        # ── Step 2: Extract sections with LlamaParse ──────────────────────────
        _emit_progress(progress_callback, "extracting", "Extracting title, abstract and content…", 35)
        extracted = extract_sections_with_llamaparse(file_bytes)

        if extracted and extracted.get("title") and len(extracted.get("content", "")) > 50:
            title = str(extracted["title"])
            abstract = str(extracted.get("abstract") or "")
            content = str(extracted["content"])
            print(f"[Pipeline] LlamaParse SUCCESS. Title: {title[:50]}")
        else:
            # Fallback: PyPDF2 → LLM extraction
            print(f"[Pipeline] LlamaParse returned insufficient data. Falling back to PyPDF2 + LLM...")
            raw_text = extract_text_from_pdf(file_bytes)
            if not raw_text:
                print(f"[Pipeline] ERROR: No text found in PDF {request.slug}")
                _post_pdf_callback({
                    "event": "failed",
                    "task_id": request.task_id,
                    "generation": request.generation,
                    "content_revision": request.content_revision,
                    "error_code": "pdf_extract_failed",
                    "retryable": False,
                })
                return {
                    "status": "error",
                    "error_code": "pdf_extract_failed",
                    "error": "Could not extract any text from PDF",
                }

            print(f"[Pipeline] Raw text extracted ({len(raw_text)} chars). Asking LLM for metadata...")
            metadata = extract_metadata_from_text(raw_text)
            title = metadata.get("title", "Untitled Paper")
            abstract = metadata.get("abstract", "No abstract available.")
            # Remove references from the raw text fallback as well
            content = remove_references(raw_text)
            print(f"[Pipeline] LLM Fallback SUCCESS. Title: {title[:50]}")

        print(f"[Pipeline] Title: {title[:80]}")
        print(f"[Pipeline] Abstract: {abstract[:120]}...")
        print(f"[Pipeline] Content: {len(content)} characters")

        # ── Step 3: Persist extracted article fields through the fenced callback
        _emit_progress(progress_callback, "saving_article", "Saving extracted article data…", 60)
        _post_pdf_callback({
            "event": "article_saved",
            "task_id": request.task_id,
            "generation": request.generation,
            "content_revision": request.content_revision,
            "title": title,
            "abstract": abstract,
            "content": content,
        })
        article_saved = True

        # ── Step 4: Chunk & Store Embeddings in PGVector ─────────────────────
        _emit_progress(progress_callback, "embedding", "Creating and storing document embeddings…", 85)
        embed_text = f"{title}\n\n{abstract}\n\n{content}"
        try:
            paper_chunks = chunk_paper(
                title=title,
                abstract=abstract,
                content=content,
                source_format="llamaparse_markdown" if extracted else "fallback_raw_text",
            )
            chunk_type_counts = {}
            for chunk in paper_chunks:
                chunk_type_counts[chunk.chunk_type] = chunk_type_counts.get(chunk.chunk_type, 0) + 1
            print(f"[Pipeline] Created {len(paper_chunks)} structured chunks: {chunk_type_counts}")
            result = _ingest_pdf_chunks(request, paper_chunks, ingest_paper_chunks)
        except Exception as chunk_error:
            logger.error(f"[Pipeline] Structured chunking failed, falling back to legacy chunk_text: {chunk_error}")
            chunks = chunk_text(embed_text)
            print(f"[Pipeline] Created {len(chunks)} legacy text chunks for embedding.")
            result = _ingest_pdf_chunks(request, chunks, ingest_article_chunks)
        print(f"[Pipeline] Vector DB ingestion result: {result}")

        if not result.get("stored"):
            error_msg = result.get("error", "Unknown embedding error")
            logger.error(f"[Pipeline] Embedding ingestion failed for article {request.slug}: {error_msg}")
            _post_pdf_callback({
                "event": "failed",
                "task_id": request.task_id,
                "generation": request.generation,
                "content_revision": request.content_revision,
                "error_code": result.get("error_code", "embedding_persist_failed"),
                "retryable": False,
            })
            return {
                "status": "partial_success",
                "error_code": result.get("error_code", "embedding_persist_failed"),
                "error": "Article extraction completed, but embedding storage failed.",
                "embedding_error": error_msg,
                "article_saved": True,
                "retryable": False,
                "title": title,
                "abstract": abstract,
                "content": content
            }

        _post_pdf_callback({
            "event": "embeddings_saved",
            "task_id": request.task_id,
            "generation": request.generation,
            "content_revision": request.content_revision,
        })

        print(f"[Pipeline] ✅ Completed for article: {request.slug}")
        return {
            "status": "success",
            "title": title,
            "abstract": abstract,
            "content": content
        }

    except Exception as e:
        logger.error("PDF pipeline failed for article %s (%s)", request.slug, type(e).__name__)
        error_code = "pdf_processing_failed"
        if isinstance(e, HTTPException):
            error_code = "pdf_callback_rejected" if e.status_code == 409 else "article_persist_failed"
        try:
            _post_pdf_callback({
                "event": "failed",
                "task_id": request.task_id,
                "generation": request.generation,
                "content_revision": request.content_revision,
                "error_code": error_code,
                "retryable": False,
            })
        except Exception:
            logger.error("Unable to persist PDF failure state for task %s", request.task_id)
        return {
            "status": "error",
            "error_code": error_code,
            "error": "PDF processing failed. Please try again.",
            "article_saved": article_saved,
        }


def _post_pdf_callback(payload: dict) -> dict:
    callback_url = f"{settings.BACKEND_API_URL.rstrip('/')}/ai/pdf/callback/"
    for attempt in range(3):
        try:
            response = requests.post(
                callback_url,
                json=payload,
                headers={
                    "X-Internal-Service-Key": settings.INTERNAL_SERVICE_KEY,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            if response.status_code == 409:
                raise HTTPException(status_code=409, detail="PDF job generation or article revision is stale")
            if response.status_code < 500:
                response.raise_for_status()
            else:
                response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise RuntimeError("Backend returned an invalid PDF callback receipt")
            return result
        except HTTPException:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code < 500:
                raise
            if attempt == 2:
                raise
            time.sleep(1 if attempt == 0 else 2)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == 2:
                raise
            time.sleep(1 if attempt == 0 else 2)
    raise RuntimeError("PDF callback retries exhausted")


def _ingest_pdf_chunks(request: PDFUploadRequest, chunks: list, ingest) -> dict:
    for attempt in range(3):
        result = ingest(
            request.article_id,
            chunks,
            pdf_generation=request.generation,
            content_revision=request.content_revision,
        )
        if result.get("stored") or not result.get("retryable") or attempt == 2:
            return result
        time.sleep(1 if attempt == 0 else 2)
    return {"stored": False, "error": "Embedding retries exhausted", "retryable": False}


@router.post("/upload")
def upload_pdf(
    request: PDFUploadRequest,
    background_tasks: BackgroundTasks,
    sync: bool = False,
    x_internal_service_key: Optional[str] = Header(default=None, alias="X-Internal-Service-Key"),
):
    """
    Triggered by the Frontend after R2 upload completes.

    Accepts the R2 public URL, article slug, and article ID,
    then either waits for the pipeline to finish (sync=True)
    or queues it as a background task (sync=False).
    
    Using 'def' (not 'async def') so FastAPI runs this in a thread pool,
    preventing the blocking of the main event loop during extraction.
    """
    _require_internal_service_key(x_internal_service_key)
    if sync:
        result = process_pdf_pipeline(request)
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        return {
            "status": "completed",
            "slug": request.slug,
            "data": result
        }
    else:
        task_id = str(uuid.UUID(request.task_id))
        if not _job_store.is_persistent:
            raise HTTPException(status_code=503, detail="PDF job storage is unavailable; refusing non-durable admission")
        try:
            created = _create_pdf_job(task_id, request)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Unable to atomically claim PDF task %s", task_id)
            raise HTTPException(status_code=503, detail="PDF job storage is unavailable") from exc
        if created:
            background_tasks.add_task(_run_pdf_job, task_id, request)
        return {
            "status": "processing_started" if created else (_job_store.get_job(task_id) or {}).get("status", "failed"),
            "task_id": task_id,
            "slug": request.slug,
            "article_id": request.article_id,
            "progress": _job_progress("queued", "PDF processing queued.", 0),
            "message": "PDF extraction and embedding pipeline queued successfully.",
        }


@router.get("/status")
async def pdf_status(
    task_id: str,
    x_internal_service_key: Optional[str] = Header(default=None, alias="X-Internal-Service-Key"),
):
    """Return the real PDF job state for the authenticated backend proxy."""
    _require_internal_service_key(x_internal_service_key)
    try:
        job = _job_store.get_pdf_job(task_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="PDF job status is temporarily unavailable") from exc
    if not job:
        raise HTTPException(
            status_code=404,
            detail="PDF task not found or expired",
            headers={"X-Error-Code": "pdf_job_not_found_or_expired"},
        )

    payload = {
        "kind": "pdf",
        "status": job.get("status", "queued"),
        "task_id": task_id,
        "slug": job.get("slug"),
        "article_id": job.get("article_id"),
        "generation": job.get("generation"),
        "content_revision": job.get("content_revision"),
        "article_saved": bool(job.get("article_saved", False)),
        "embeddings_saved": bool(job.get("embeddings_saved", False)),
        "progress": job.get(
            "progress",
            _job_progress("queued", "PDF processing queued.", 0),
        ),
    }
    if job.get("status") == "failed":
        payload.update(
            {
                "error_code": job.get("error_code", "pdf_processing_failed"),
                "message": job.get("message", "PDF processing failed."),
                "retryable": bool(job.get("retryable", True)),
            }
        )
    return payload
