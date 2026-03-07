from typing import Any

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
	session_id: str | None = None
	article_id: int
	answer: str
	citations: list[dict[str, Any]] = Field(default_factory=list)
	confidence_score: float
	review_feedback: str | None = None
	need_clarification: bool = False
	clarification_question: str | None = None
