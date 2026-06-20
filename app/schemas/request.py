from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
	question: str = Field(min_length=3)
	article_id: int | None = None
	session_id: str | None = None
	debug: bool = False

	@model_validator(mode="before")
	@classmethod
	def accept_message_alias(cls, data):
		if isinstance(data, dict) and "question" not in data and "message" in data:
			return {**data, "question": data["message"]}
		return data


class ResearchRequest(BaseModel):
	query: str = Field(min_length=3)
	debug: bool = False

	@model_validator(mode="before")
	@classmethod
	def accept_message_alias(cls, data):
		if isinstance(data, dict) and "query" not in data and "message" in data:
			return {**data, "query": data["message"]}
		return data
