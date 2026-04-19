import os
import urllib.parse

import boto3
import requests
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from app.core.config import settings
from app.core.logger import get_logger
from app.pdf_pipeline.llama_extractor import extract_sections_with_llamaparse, remove_references
from app.pdf_pipeline.extractor import extract_text_from_pdf
from app.pdf_pipeline.llm_extractor import extract_metadata_from_text
from app.pdf_pipeline.chunker import chunk_text
from app.embeddings.vector_store import ingest_article_chunks

router = APIRouter()
logger = get_logger(__name__)


class PDFUploadRequest(BaseModel):
    pdf_url: str
    slug: str
    article_id: int


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
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    file_bytes = response['Body'].read()
    print(f"Downloaded {len(file_bytes)} bytes from Cloudflare R2.")
    return file_bytes


def process_pdf_pipeline(request: PDFUploadRequest):
    """
    Full AI extraction pipeline.

    Flow:
    1. Download PDF from Cloudflare R2 via boto3
    2. Extract title/abstract/content via LlamaParse (IEEE-aware)
       → Falls back to PyPDF2 + LLM if LlamaParse is unavailable
    3. PUT extracted fields to Django Backend
    4. Chunk content and store in PGVector
    """
    print(f"[Pipeline] Starting for article: {request.slug}")
    try:
        # ── Step 1: Download PDF ──────────────────────────────────────────────
        file_bytes = _download_pdf_from_r2(request.pdf_url)

        # ── Step 2: Extract sections with LlamaParse ──────────────────────────
        extracted = extract_sections_with_llamaparse(file_bytes)

        if extracted and extracted.get("title") and len(extracted.get("content", "")) > 50:
            title = extracted["title"]
            abstract = extracted["abstract"]
            content = extracted["content"]
            print(f"[Pipeline] LlamaParse SUCCESS. Title: {title[:50]}")
        else:
            # Fallback: PyPDF2 → LLM extraction
            print(f"[Pipeline] LlamaParse returned insufficient data. Falling back to PyPDF2 + LLM...")
            raw_text = extract_text_from_pdf(file_bytes)
            if not raw_text:
                print(f"[Pipeline] ERROR: No text found in PDF {request.slug}")
                return {"error": "Could not extract any text from PDF"}

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

        # ── Step 3: Update Django Backend via internal API ────────────────────
        # Ensure the URL ends with a trailing slash for Django
        backend_url = settings.BACKEND_API_URL.rstrip('/')
        update_url = f"{backend_url}/articles/{request.slug}/"
        
        update_payload = {
            "title": title,
            "abstract": abstract,
            "content": content,
            "pdf_url": request.pdf_url,
        }

        backend_headers = {
            "X-Internal-Service-Key": settings.INTERNAL_SERVICE_KEY,
            "Content-Type": "application/json"
        }

        print(f"[Pipeline] Patching Backend: {update_url}")
        backend_response = requests.patch(
            update_url,
            json=update_payload,
            headers=backend_headers,
            timeout=30
        )

        if backend_response.ok:
            print(f"[Pipeline] Updated Django Backend successfully (HTTP {backend_response.status_code}).")
        else:
            print(f"[Pipeline] WARNING: Backend update returned {backend_response.status_code}: {backend_response.text}")

        # ── Step 4: Chunk & Store Embeddings in PGVector ─────────────────────
        embed_text = f"{title}\n\n{abstract}\n\n{content}"
        chunks = chunk_text(embed_text)
        print(f"[Pipeline] Created {len(chunks)} text chunks for embedding.")

        result = ingest_article_chunks(request.article_id, chunks)
        print(f"[Pipeline] Vector DB ingestion result: {result}")

        if not result.get("stored"):
            error_msg = result.get("error", "Unknown embedding error")
            logger.error(f"[Pipeline] Embedding ingestion failed for article {request.slug}: {error_msg}")
            return {
                "status": "partial_success",
                "embedding_error": error_msg,
                "title": title,
                "abstract": abstract,
                "content": content
            }

        print(f"[Pipeline] ✅ Completed for article: {request.slug}")
        return {
            "status": "success",
            "title": title,
            "abstract": abstract,
            "content": content
        }

    except Exception as e:
        print(f"[Pipeline] ❌ Error for article {request.slug}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@router.post("/upload")
def upload_pdf(
    request: PDFUploadRequest,
    background_tasks: BackgroundTasks,
    sync: bool = False
):
    """
    Triggered by the Frontend after R2 upload completes.

    Accepts the R2 public URL, article slug, and article ID,
    then either waits for the pipeline to finish (sync=True)
    or queues it as a background task (sync=False).
    
    Using 'def' (not 'async def') so FastAPI runs this in a thread pool,
    preventing the blocking of the main event loop during extraction.
    """
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
        background_tasks.add_task(process_pdf_pipeline, request)
        return {
            "status": "processing_started",
            "slug": request.slug,
            "message": "LlamaParse extraction + vector embedding pipeline queued successfully."
        }


@router.get("/status")
async def pdf_status(slug: str):
    """
    Placeholder endpoint for pipeline status polling.
    Real-time tracking to be implemented with Redis/Celery later.
    """
    return {
        "status": "pending_or_completed",
        "slug": slug,
        "message": "Real-time task polling to be implemented via Redis/RabbitMQ later."
    }
