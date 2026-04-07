"""
LLM factory — tách riêng 2 LLM:
- agent_llm: Groq (llama-3.1-8b-instant) — dùng cho tất cả agents
- extract_llm: Google/OpenAI — dùng cho pdf_pipeline (giữ nguyên)
"""
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings


def get_agent_llm():
    """LLM cho agents — mặc định Groq."""
    if settings.AGENT_LLM_PROVIDER == "groq":
        return ChatGroq(
            model=settings.GROQ_LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
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
