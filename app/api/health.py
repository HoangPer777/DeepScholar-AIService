from fastapi import APIRouter

from app.embeddings import vector_store

router = APIRouter()

@router.get("/")
def health_check():
    return {"status": "ok", "database": vector_store.database_health()}
