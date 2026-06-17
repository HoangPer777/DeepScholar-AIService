from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode


from typing import Annotated, List
from pydantic import field_validator


class Settings(BaseSettings):
    PROJECT_NAME: str = "DeepScholar AI"
    API_PREFIX: str = "/api"
    OPENAI_API_KEY: str = ""
    DATABASE_URL: str
    CORS_ALLOW_ORIGINS: Annotated[List[str], NoDecode] = ["http://localhost:3000"]
    UPLOAD_DIR: str = str(Path(__file__).resolve().parents[2] / "data" / "uploads")

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    EMBEDDING_PROVIDER: str = "google"  # Options: "openai", "google", "huggingface"
    EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 3072  # Read from .env; must match the model output size
    GOOGLE_EMBEDDING_MODEL: str = "models/gemini-embedding-2-preview"
    GOOGLE_API_KEY: str = ""
    GOOGLE_EMBEDDING_MAX_RETRIES: int = 6
    GOOGLE_EMBEDDING_RETRY_BASE_SECONDS: float = 2.0
    GOOGLE_EMBEDDING_REQUEST_DELAY_SECONDS: float = 1.0
    CHUNKING_STRATEGY: str = "paper_structure"
    CHUNKING_VERSION: str = "v2"
    CHUNK_TARGET_TOKENS: int = 700
    CHUNK_MAX_TOKENS: int = 950
    CHUNK_OVERLAP_TOKENS: int = 100
    TABLE_MAX_ROWS_PER_CHUNK: int = 20
    MAX_CHAT_HISTORY: int = 20

    # LlamaParse API key from https://cloud.llamaindex.ai
    LLAMAPARSE_API_KEY: str = ""

    # Django backend base URL for internal service calls
    BACKEND_API_URL: str = "http://backend:8000/api/v1"

    # Internal secret for AI service -> Backend communication
    INTERNAL_SERVICE_KEY: str

    # ============================================================
    # AGENT LLM (Groq)
    # ============================================================
    GROQ_API_KEY: str = ""
    TAVILY_API_KEY: str = ""
    GROQ_LLM_MODEL: str = "llama-3.1-8b-instant"
    AGENT_LLM_PROVIDER: str = "groq"  # "groq" | "openai" | "google"

    # ============================================================
    # OPENROUTER (Multi-Model Support — V12)
    # ============================================================
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # ============================================================
    # REDIS (Short-Term Memory — Memory Chatbot)
    # ============================================================
    REDIS_URL: str = "redis://localhost:6379/0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

