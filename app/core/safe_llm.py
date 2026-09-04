"""
SafeLLM — OpenRouter multi-model router.

Tries OpenRouter free-model candidates in order. Groq fallback is optional and
disabled by default for OpenRouter-only deployments.
"""
import time
from typing import Any, List, Optional

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
        self._attempts: list[dict[str, Any]] = []
        self._selected_model: Optional[str] = None
        self._selected_provider: Optional[str] = None
        self._fallback_used = False
        self._invocations = 0

    def invoke(self, messages: List) -> AIMessage:
        """Try OpenRouter candidates in order, then optional Groq fallback."""
        self._attempts = []
        self._selected_model = None
        self._selected_provider = None
        self._fallback_used = False
        self._invocations += 1
        failures: list[str] = []

        if not settings.OPENROUTER_API_KEY:
            if settings.ENABLE_GROQ_FALLBACK and self.groq_fallback is not None:
                return self._invoke_groq_fallback(messages)
            raise AllLLMProvidersFailed(
                f"OpenRouter API key is missing for agent '{self.agent_name}'"
            )

        for model_name in self.candidates:
            started = time.perf_counter()
            try:
                print(f"\n[SafeLLM:{self.agent_name}] Trying: {model_name}")
                llm = self._build_openrouter_llm(model_name)
                response = llm.invoke(messages)
                print(f"[SafeLLM:{self.agent_name}] SUCCESS: {model_name}")
                self._attempts.append(
                    {
                        "model": model_name,
                        "provider": "OpenRouter",
                        "status": "success",
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                self._selected_model = model_name
                self._selected_provider = "OpenRouter"
                return response
            except Exception as exc:
                error = str(exc)[:240]
                error_type = self._classify_error(exc)
                failures.append(f"{model_name}: {error_type}")
                self._attempts.append(
                    {
                        "model": model_name,
                        "provider": "OpenRouter",
                        "status": "failed",
                        "error_type": error_type,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                print(f"[SafeLLM:{self.agent_name}] FAILED: {model_name}")
                print(f"  Error type: {error_type}")
                print(f"  Error: {error}")
                if error_type == "auth_error":
                    break
                time.sleep(0.5)

        if settings.ENABLE_GROQ_FALLBACK and self.groq_fallback is not None:
            print(f"\n[SafeLLM:{self.agent_name}] All OpenRouter candidates failed, using Groq fallback")
            self._fallback_used = True
            return self._invoke_groq_fallback(messages)

        raise AllLLMProvidersFailed(
            f"All OpenRouter free models failed for agent '{self.agent_name}': "
            + "; ".join(failures)
        )

    def _invoke_groq_fallback(self, messages: List) -> AIMessage:
        started = time.perf_counter()
        fallback_model = getattr(self.groq_fallback, "model_name", None) or getattr(
            self.groq_fallback, "model", None
        )
        if not isinstance(fallback_model, str) or not fallback_model.strip():
            fallback_model = "configured-groq-model"
        try:
            response = self.groq_fallback.invoke(messages)
            print(f"[SafeLLM:{self.agent_name}] SUCCESS: Groq fallback")
            self._attempts.append(
                {
                    "model": str(fallback_model),
                    "provider": "Groq",
                    "status": "success",
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                }
            )
            self._selected_model = str(fallback_model)
            self._selected_provider = "Groq"
            return response
        except Exception as exc:
            print(f"[SafeLLM:{self.agent_name}] FAILED: Groq fallback")
            print(f"  Error: {str(exc)[:200]}")
            self._attempts.append(
                {
                    "model": str(fallback_model),
                    "provider": "Groq",
                    "status": "failed",
                    "error_type": self._classify_error(exc),
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                }
            )
            raise AllLLMProvidersFailed(
                f"All LLM providers failed for agent '{self.agent_name}'"
            ) from exc

    def usage_snapshot(self) -> dict[str, Any]:
        """Return safe model routing telemetry for the research response."""
        if self._selected_model:
            status = "selected"
            model = self._selected_model
            provider = self._selected_provider or "Unknown"
        elif self._attempts:
            status = "failed"
            model = None
            provider = str(self._attempts[-1].get("provider") or "Unknown")
        else:
            status = "not_called"
            model = None
            provider = "Not called"

        return {
            "agent": self.agent_name,
            "provider": provider,
            "model": model,
            "selected_provider": provider,
            "selected_model": model,
            "status": status,
            "routing": "Groq fallback" if self._fallback_used else "OpenRouter candidates",
            "fallback_used": self._fallback_used,
            "invocations": self._invocations,
            "available_models": list(self.candidates),
            "attempts": list(self._attempts),
        }

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
