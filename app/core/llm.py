import os
from functools import lru_cache


def _resolve_provider() -> str:
    configured = os.getenv("LLM_PROVIDER", "").strip().lower()
    if configured:
        return configured

    # Auto-pick provider based on configured keys when LLM_PROVIDER is unset.
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.getenv("GOOGLE_API_KEY", "").strip():
        return "gemini"
    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"

    # Keep backward-compatible default, but error will explain missing keys.
    return "groq"


def _build_llm(provider: str, timeout_s: float, retries: int):
    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.1,
            api_key=api_key,
            max_retries=retries,
            timeout=timeout_s,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=retries,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0.1,
            google_api_key=api_key,
            timeout=timeout_s,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            temperature=0.1,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=retries,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache(maxsize=1)
def get_llm():
    provider = _resolve_provider()
    timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
    retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
    provider_order = [provider, "openai", "gemini", "groq", "anthropic"]

    attempted = []
    for p in dict.fromkeys(provider_order):
        try:
            return _build_llm(p, timeout_s=timeout_s, retries=retries)
        except Exception as exc:
            attempted.append(f"{p}: {exc}")

    raise RuntimeError("Cannot initialize any LLM provider. Attempts: " + " | ".join(attempted))


@lru_cache(maxsize=1)
def get_fast_llm():
    provider = _resolve_provider()
    timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
    retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
                temperature=0.1,
                api_key=os.getenv("GROQ_API_KEY"),
                max_retries=retries,
                timeout=timeout_s,
            )
        except Exception:
            return get_llm()
    return get_llm()