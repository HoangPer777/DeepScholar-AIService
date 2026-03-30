from pydantic import AliasChoices, BaseModel, Field


class ChatRequest(BaseModel):
	question: str = Field(min_length=3)
	article_id: int
	session_id: str | None = None


class ResearchRequest(BaseModel):
	question: str = Field(min_length=3, validation_alias=AliasChoices("question", "query"))
	article_id: int | None = None
	max_iterations: int = Field(default=2, ge=1, le=5)
	session_id: str | None = None
