from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chatbot, health, pdf, research
from app.core.config import settings
from app.core.graph_checkpoint import close_graph_checkpointer, get_graph_checkpointer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_graph_checkpointer()
    try:
        yield
    finally:
        close_graph_checkpointer()


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(pdf.router, prefix=f"{settings.API_PREFIX}/pdf", tags=["PDF Pipeline"])
app.include_router(chatbot.router, prefix=f"{settings.API_PREFIX}/chat", tags=["Chatbot"])
app.include_router(research.router, prefix=f"{settings.API_PREFIX}/research", tags=["Research"])


@app.get("/")
def read_root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}
