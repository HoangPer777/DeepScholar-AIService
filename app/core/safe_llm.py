"""
SafeLLM — OpenRouter multi-model router.

Tries OpenRouter free-model candidates in order. Groq fallback is optional and
disabled by default for OpenRouter-only deployments.
"""
import time
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


class AllLLMProvidersFailed(RuntimeError):
    """Raised when every configured LLM candidate fails."""


class SafeLLM:
    """Multi-model LLM router with ordered fallback."""

    def __init__(
        self,
        agent_name: str,
        candidates: List[str],
        groq_fallback: Optional[BaseChatModel] = None,
    ):
        self.agent_name = agent_name
        self.candidates = candidates
        self.groq_fallback = groq_fallback

    def invoke(self, messages: List) -> AIMessage:
        """Try OpenRouter candidates in order, then optional Groq fallback."""
        failures: list[str] = []

        if not settings.OPENROUTER_API_KEY:
            if settings.ENABLE_GROQ_FALLBACK and self.groq_fallback is not None:
                return self._invoke_groq_fallback(messages)
            raise AllLLMProvidersFailed(
                f"OpenRouter API key is missing for agent '{self.agent_name}'"
            )

        for model_name in self.candidates:
            try:
                print(f"\n[SafeLLM:{self.agent_name}] Trying: {model_name}")
                llm = self._build_openrouter_llm(model_name)
                response = llm.invoke(messages)
                print(f"[SafeLLM:{self.agent_name}] SUCCESS: {model_name}")
                return response
            except Exception as exc:
                error = str(exc)[:240]
                error_type = self._classify_error(exc)
                failures.append(f"{model_name}: {error_type}")
                print(f"[SafeLLM:{self.agent_name}] FAILED: {model_name}")
                print(f"  Error type: {error_type}")
                print(f"  Error: {error}")
                if error_type == "auth_error":
                    break
                time.sleep(0.5)

        if settings.ENABLE_GROQ_FALLBACK and self.groq_fallback is not None:
            print(f"\n[SafeLLM:{self.agent_name}] All OpenRouter candidates failed, using Groq fallback")
            return self._invoke_groq_fallback(messages)

        raise AllLLMProvidersFailed(
            f"All OpenRouter free models failed for agent '{self.agent_name}': "
            + "; ".join(failures)
        )

    def _invoke_groq_fallback(self, messages: List) -> AIMessage:
        try:
            response = self.groq_fallback.invoke(messages)
            print(f"[SafeLLM:{self.agent_name}] SUCCESS: Groq fallback")
            return response
        except Exception as exc:
            print(f"[SafeLLM:{self.agent_name}] FAILED: Groq fallback")
            print(f"  Error: {str(exc)[:200]}")
            raise AllLLMProvidersFailed(
                f"All LLM providers failed for agent '{self.agent_name}'"
            ) from exc

    def _build_openrouter_llm(self, model: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            temperature=0,
            max_retries=0,
            timeout=settings.OPENROUTER_MODEL_TIMEOUT_SECONDS,
            default_headers={
                "HTTP-Referer": "https://deepscholar.ai",
                "X-Title": "DeepScholar Deep Research V12",
            },
        )

    def _classify_error(self, exc: Exception) -> str:
        text = str(exc).lower()
        if "429" in text or "rate" in text:
            return "rate_limited"
        if "401" in text or "403" in text or "unauthorized" in text or "api key" in text:
            return "auth_error"
        if "404" in text or "not found" in text:
            return "model_not_found"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if "context" in text and "length" in text:
            return "context_error"
        if "500" in text or "502" in text or "503" in text or "504" in text:
            return "provider_unavailable"
        return "unknown"
