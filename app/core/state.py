from typing import List, Dict, Optional, Annotated
import operator
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    question: str

    need_clarification: bool = False
    need_external_search: bool = True
    research_queries: List[str] = []

    vector_context: Annotated[List[Dict], operator.add] = []
    external_context: Annotated[List[Dict], operator.add] = []
    memory_context: Annotated[List[Dict], operator.add] = []

    ranked_context: List[Dict] = []

    clarified_question: Optional[str] = None
    draft_answer: Optional[str] = None
    final_answer: Optional[str] = None

    confidence_score: float = 0.0
    feedback: str = ""
    rewrite_required: bool = False
    iteration_count: int = 0
    max_iterations: int = 2

    logs: Annotated[List[Dict], operator.add] = []

class PlanOutput(BaseModel):
    need_clarification: bool = Field(description="Câu hỏi có mơ hồ không?")
    need_external_search: bool = Field(description="Có cần tìm kiếm web không?")
    queries: List[str] = Field(description="3-5 research queries cụ thể")

class ClarifyOutput(BaseModel):
    clarified_question: str = Field(description="Câu hỏi đã được làm rõ")

class ReviewOutput(BaseModel):
    confidence_score: float = Field(ge=0.0, le=1.0, description="Điểm chất lượng 0.0-1.0")
    feedback: str = Field(description="Nhận xét cụ thể để writer cải thiện")
    rewrite_required: bool = Field(description="Có cần viết lại không?")