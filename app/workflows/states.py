from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    question: str
    article_id: int
    session_id: Optional[str] = None

    need_clarification: bool = False
    need_external_search: bool = False
    focus_sections: List[str] = Field(default_factory=list)

    vector_context: List[Dict[str, Any]] = Field(default_factory=list)
    external_context: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)

    draft_answer: Optional[str] = None
    reviewed_answer: Optional[str] = None
    clarification_question: Optional[str] = None

    review_feedback: Optional[str] = None
    confidence_score: float = 0.0
    iteration_count: int = 0
    max_iterations: int = 2

    memory_id: Optional[str] = None
