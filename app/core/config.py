from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "DeepScholar AI"
    API_PREFIX: str = "/api"
    OPENAI_API_KEY: str = ""
    DATABASE_URL: str = "postgresql://deepscholar:deepscholar@localhost:5432/deepscholar"
    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    UPLOAD_DIR: str = str(Path(__file__).resolve().parents[2] / "data" / "uploads")

    EMBEDDING_PROVIDER: str = "openai"  # Options: "openai" or "google"
    GOOGLE_API_KEY: str = ""
    MAX_CHAT_HISTORY: int = 20

    # LlamaParse API key from https://cloud.llamaindex.ai
    LLAMAPARSE_API_KEY: str = ""

    # Django backend base URL for internal service calls
    BACKEND_API_URL: str = "http://backend:8000/api/v1"

    # Internal secret for AI service -> Backend communication
    INTERNAL_SERVICE_KEY: str = "deepscholar-secret-key-2026"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def EMBEDDING_DIMENSION(self) -> int:
        # gemini-embedding-001 in this project returns 3072 dimensions
        return 3072 if self.EMBEDDING_PROVIDER == "google" else 1536


settings = Settings()

