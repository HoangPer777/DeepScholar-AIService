from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    # User input
    question: str
    article_id: Optional[int] = None

    # Decisions (from Planner)
    need_clarification: bool = False
    need_external_search: bool = False
    focus_sections: List[str] = Field(default_factory=list)

    # Context (Retrieved data)
    vector_context: List[Dict] = Field(default_factory=list)
    external_context: List[Dict] = Field(default_factory=list)

    # Outputs
    draft_answer: Optional[str] = None
    reviewed_answer: Optional[str] = None

    # Control metrics (Review & control)
    confidence_score: float = 0.0
    iteration_count: int = 0
    max_retries: int = 2
    review_feedback: Optional[str] = None
    
    # Memory
    memory_id: Optional[str] = None

def should_rewrite(state: AgentState) -> str:
    """
    Loop condition logic: dictates whether to rewrite the answer or accept it.
    """
    if state.confidence_score < 0.7 and state.iteration_count < state.max_retries:
        return "rewrite"
    return "accept"
