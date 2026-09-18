import operator
from typing import Annotated, Dict, List, Optional

from pydantic import BaseModel, Field


def _merge_dicts(left: Dict, right: Dict) -> Dict:
    """Merge parallel LangGraph dictionary updates without mutating inputs."""
    return {**left, **right}


class AgentState(BaseModel):
    question: str

    article_id: Optional[int] = None

    need_clarification: bool = False
    need_external_search: bool = True
    need_internal_search: bool = True

    # Những section nào của paper quan trọng nhất?
    focus_sections: List[str] = Field(default_factory=list)
    # Danh sách queries tìm kiếm web
    search_queries: List[str] = Field(default_factory=list)
    web_search_queries: List[str] = Field(default_factory=list)
    db_search_queries: List[str] = Field(default_factory=list)
    search_keywords: List[str] = Field(default_factory=list)

    # Lưu câu hỏi sau khi làm rõ
    clarified_question: Optional[str] = None

    # Lưu chunks từ PDF/bài viết upload
    vector_context: List[Dict] = Field(default_factory=list)
    # Lưu kết quả từ web search
    external_context: List[Dict] = Field(default_factory=list)
    reader_status: str = "pending"
    researcher_status: str = "pending"
    retrieval_warnings: Annotated[List[str], operator.add] = Field(default_factory=list)
    evidence_manifest: List[Dict] = Field(default_factory=list)
    evidence_available: bool = False
    workflow_status: str = "pending"
    review_decision: Optional[str] = None
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None
    checkpoint_thread_id: Optional[str] = None

    # Bài viết lần đầu tiên từ Writer
    draft_answer: Optional[str] = None
    # Bài viết đã được phê bình/phê duyệt
    reviewed_answer: Optional[str] = None

    # Lưu feedback từ Reviewer để Writer cải thiện
    review_feedback: Optional[str] = None
    # Artifacts exposed to the initiating user for the Deep Research trace.
    draft_history: List[Dict] = Field(default_factory=list)
    review_history: List[Dict] = Field(default_factory=list)

    # Điểm số chất lượng (0.0 - 1.0)
    confidence_score: float = 0.0
    # Số lần Writer đã viết lại
    iteration_count: int = 0
    # Giới hạn số lần viết lại
    max_iterations: int = 2

    # To track agent execution flow
    logs: Annotated[List[str], operator.add] = Field(default_factory=list)
    timings: Annotated[Dict[str, int], operator.ior] = Field(default_factory=dict)
    model_usage: Annotated[Dict[str, Dict], _merge_dicts] = Field(default_factory=dict)
    writer_model: Optional[Dict] = None

    # NEW: timing metadata for observability (Requirement 8.1)
    # Populated by the workflow routing layer, not the agents themselves
    planner_time: float = 0.0
    researcher_time: float = 0.0
    writer_time: float = 0.0
    reviewer_time: float = 0.0
