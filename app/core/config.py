from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "DeepScholar AI Service"
    API_PREFIX: str = "/api/v1"

    ENV: str = "dev"
    DEBUG: bool = False
    PORT: int = 8000

    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ORIGINS: str = "*"

    DATABASE_URL: str = "postgresql://deepscholar:deepscholar@localhost:5432/deepscholar"

    EMBEDDING_PROVIDER: str = "google"
    EMBEDDING_DIMENSION: int = 768
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""

    BACKEND_API_URL: str = "http://host.docker.internal:8000/api/v1"
    INTERNAL_SERVICE_KEY: str = ""

    LLM_PROVIDER: str = "groq"
    REQUEST_TIMEOUT_SECONDS: int = 45

    RATE_LIMIT_PER_MINUTE: int = 10

    @property
    def cors_allow_origins_list(self) -> List[str]:
        raw = (self.CORS_ALLOW_ORIGINS or self.CORS_ORIGINS or "*").strip()
        if not raw:
            return ["*"]

        # Accept both JSON-like list syntax and comma-separated plain strings from .env
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]

        origins = [item.strip().strip('"').strip("'") for item in raw.split(",") if item.strip()]
        return origins or ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
