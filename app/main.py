from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chatbot, health, pdf, research
from app.core.config import settings


app = FastAPI(title=settings.PROJECT_NAME)

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
