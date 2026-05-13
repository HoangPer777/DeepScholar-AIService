"""
LLM factory — tách riêng 2 LLM:
- agent_llm: Groq (llama-3.3-70b-versatile) — dùng cho tất cả agents
- extract_llm: Google/OpenAI — dùng cho pdf_pipeline (giữ nguyên)

V12 Update:
- get_safe_llm(agent_name): Multi-model router với OpenRouter + Groq fallback
- get_agent_llm(): Giữ nguyên cho backward compatibility
"""
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.model_candidates import MODEL_CANDIDATES
from app.core.safe_llm import SafeLLM


def get_agent_llm():
    """LLM cho agents — mặc định Groq với max_retries để tự xử lý 429."""
    if settings.AGENT_LLM_PROVIDER == "groq":
        return ChatGroq(
            model=settings.GROQ_LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_retries=3,  # tự retry khi gặp 429, với exponential backoff
        )
    elif settings.AGENT_LLM_PROVIDER == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,
        )
    else:
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )


def get_safe_llm(agent_name: str) -> BaseChatModel:
    """
    V12 Multi-model router với auto-fallback.

    Preconditions:
    - agent_name là valid key trong MODEL_CANDIDATES

    Postconditions:
    - Nếu OPENROUTER_API_KEY có → trả về SafeLLM với candidates
    - Nếu không → trả về get_agent_llm() như cũ (backward compatible)

    Args:
        agent_name: Tên agent ("planner", "clarifier", "researcher", "writer", "reviewer")

    Returns:
        SafeLLM instance hoặc BaseChatModel (Groq/OpenAI/Google)
    """
    # Nếu không có OPENROUTER_API_KEY → fallback về get_agent_llm() như V11
    if not settings.OPENROUTER_API_KEY:
        return get_agent_llm()

    # Lấy candidates cho agent này
    candidates = MODEL_CANDIDATES.get(agent_name, [])
    if not candidates:
        # Nếu agent_name không có trong MODEL_CANDIDATES → fallback về get_agent_llm()
        return get_agent_llm()

    # Tạo Groq fallback
    groq_fallback = ChatGroq(
        model=settings.GROQ_LLM_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
        max_retries=3,
    )

    # Trả về SafeLLM với candidates và Groq fallback
    return SafeLLM(
        agent_name=agent_name,
        candidates=candidates,
        groq_fallback=groq_fallback,
    )
