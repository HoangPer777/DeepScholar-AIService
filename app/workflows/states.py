from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    question: str

    article_id: Optional[int] = None

    need_clarification: bool = False
    need_external_search: bool = False

    # Những section nào của paper quan trọng nhất?
    focus_sections: List[str] = Field(default_factory=list)
    # Danh sách queries tìm kiếm web
    search_queries: List[str] = Field(default_factory=list)

    # Lưu câu hỏi sau khi làm rõ
    clarified_question: Optional[str] = None

    # Lưu chunks từ PDF/bài viết upload
    vector_context: List[Dict] = Field(default_factory=list)
    # Lưu kết quả từ web search
    external_context: List[Dict] = Field(default_factory=list)

    # Bài viết lần đầu tiên từ Writer
    draft_answer: Optional[str] = None
    # Bài viết đã được phê bình/phê duyệt
    reviewed_answer: Optional[str] = None

    # Lưu feedback từ Reviewer để Writer cải thiện
    review_feedback: Optional[str] = None

    # Điểm số chất lượng (0.0 - 1.0)
    confidence_score: float = 0.0
    # Số lần Writer đã viết lại
    iteration_count: int = 0
    # Giới hạn số lần viết lại
    max_iterations: int = 3

    # To track agent execution flow
    logs: List[str] = Field(default_factory=list)
