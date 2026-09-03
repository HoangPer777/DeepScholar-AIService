"""
Model candidates per agent — OpenRouter free models.

Update this file when `scripts/check_openrouter_free_models.py` finds a better
free-model set. SafeLLM tries models in order and moves to the next one when a
model is rate-limited or unavailable.
"""

_STRONG_FREE_MODELS = [
    # "deepseek/deepseek-v3.2",
    # "openai/gpt-oss-120b",
    "minimax/minimax-m3:free",

    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",

    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/free",
]

_FAST_FREE_MODELS = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "poolside/laguna-xs-2.1:free",
    "openai/gpt-oss-20b:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "openrouter/free",
]

MODEL_CANDIDATES: dict[str, list[str]] = {
    "planner": _STRONG_FREE_MODELS,
    "clarifier": _FAST_FREE_MODELS,
    "researcher": _STRONG_FREE_MODELS,
    "writer": _STRONG_FREE_MODELS,
    "reviewer": _STRONG_FREE_MODELS,
    "fast_chat": _FAST_FREE_MODELS,
}
