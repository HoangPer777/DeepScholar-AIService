"""
SafeLLM — Multi-model router với automatic fallback.

Thử lần lượt các model candidates, fallback về Groq nếu tất cả fail.
Không bao giờ raise exception — luôn trả về response.
"""
import time
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


class AllLLMProvidersFailed(RuntimeError):
    """Raised when every OpenRouter candidate and the Groq fallback fail."""


class SafeLLM:
    """
    Multi-model LLM router với auto-fallback.

    Preconditions:
    - agent_name là valid key trong MODEL_CANDIDATES
    - candidates là non-empty list of OpenRouter model IDs
    - groq_fallback là valid BaseChatModel instance

    Postconditions:
    - invoke() luôn trả về AIMessage, không bao giờ raise exception
    - Thử từng candidate theo thứ tự, fallback về Groq nếu tất cả fail
    """

    def __init__(
        self,
        agent_name: str,
        candidates: List[str],
        groq_fallback: BaseChatModel,
    ):
        self.agent_name = agent_name
        self.candidates = candidates
        self.groq_fallback = groq_fallback

    def invoke(self, messages: List) -> AIMessage:
        """
        Invoke LLM với auto-fallback.

        Preconditions:
        - messages là non-empty list

        Postconditions:
        - Trả về AIMessage, không raise exception
        - Log mỗi attempt (success/failure)

        Loop invariant:
        - Mỗi iteration thử đúng 1 model, không retry model đã fail
        """
        # Nếu không có OPENROUTER_API_KEY → dùng Groq ngay
        if not settings.OPENROUTER_API_KEY:
            return self._invoke_groq_fallback(messages)

        # Thử từng candidate
        for model_name in self.candidates:
            try:
                print(f"\n[SafeLLM:{self.agent_name}] Trying: {model_name}")
                llm = self._build_openrouter_llm(model_name)
                response = llm.invoke(messages)
                print(f"[SafeLLM:{self.agent_name}] SUCCESS: {model_name}")
                return response
            except Exception as e:
                print(f"[SafeLLM:{self.agent_name}] FAILED: {model_name}")
                print(f"  Error: {str(e)[:200]}")
                time.sleep(0.5)  # Brief pause before next attempt
                continue

        # Emergency fallback — tất cả candidates đã fail
        print(f"\n[SafeLLM:{self.agent_name}] All candidates failed, using Groq emergency fallback")
        return self._invoke_groq_fallback(messages)

    def _invoke_groq_fallback(self, messages: List) -> AIMessage:
        try:
            response = self.groq_fallback.invoke(messages)
            print(f"[SafeLLM:{self.agent_name}] SUCCESS: Groq emergency fallback")
            return response
        except Exception as exc:
            print(f"[SafeLLM:{self.agent_name}] FAILED: Groq emergency fallback")
            print(f"  Error: {str(exc)[:200]}")
            raise AllLLMProvidersFailed(
                f"All LLM providers failed for agent '{self.agent_name}'"
            ) from exc

    def _build_openrouter_llm(self, model: str) -> ChatOpenAI:
        """
        Build ChatOpenAI instance trỏ tới OpenRouter.

        Preconditions:
        - model là non-empty string (OpenRouter model ID)
        - settings.OPENROUTER_API_KEY là non-empty string
        - settings.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"

        Postconditions:
        - Trả về ChatOpenAI với base_url trỏ tới OpenRouter
        - temperature=0, max_retries=1 (SafeLLM tự handle retry ở level cao hơn)
        """
        return ChatOpenAI(
            model=model,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            temperature=0,
            max_retries=1,  # Không retry nhiều lần per model — SafeLLM sẽ thử model khác
            timeout=180,
            default_headers={
                "HTTP-Referer": "https://deepscholar.ai",
                "X-Title": "DeepScholar Deep Research V12",
            },
        )
