from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "DeepScholar AI"
    API_PREFIX: str = "/api"
    OPENAI_API_KEY: str = ""
    DATABASE_URL: str = "postgresql://deepscholar:deepscholar@localhost:5432/deepscholar"
    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    UPLOAD_DIR: str = str(Path(__file__).resolve().parents[2] / "data" / "uploads")
    EMBEDDING_DIMENSION: int = 8
    MAX_CHAT_HISTORY: int = 20

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
