from fastapi import APIRouter

from app.embeddings.vector_store import database_health

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "ok", "database": database_health()}
