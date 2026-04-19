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

    EMBEDDING_PROVIDER: str = "openai"  # Options: "openai", "google", "huggingface"
    EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768  # Read from .env; must match the model output size
    GOOGLE_API_KEY: str = ""
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

