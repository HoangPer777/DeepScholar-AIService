from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter()

@router.post("/upload")
async def upload_pdf(article_id: int = Form(...), file: UploadFile = File(...)):
    """
    TODO: Handle PDF upload and processing
    1. Validate file format (PDF extension)
    2. Save file to storage
    3. Extract text from PDF
    4. Infer metadata from content
    5. Split into chunks
    6. Generate embeddings and ingest to vector store
    7. Return task_id for status tracking
    """
    # TODO: Implementation
    return {
        "task_id": "",
        "status": "pending",
        "filename": file.filename,
        "article_id": article_id
    }


@router.get("/{task_id}/status")
async def pdf_status(task_id: str):
    """
    TODO: Check PDF processing status by task_id
    """
    # TODO: Implementation
    return {
        "task_id": task_id,
        "status": "pending",
        "progress": 0
    }
