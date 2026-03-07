from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
	question: str = Field(min_length=3)
	article_id: int
	session_id: str | None = None


class ResearchRequest(BaseModel):
	query: str = Field(min_length=3)
