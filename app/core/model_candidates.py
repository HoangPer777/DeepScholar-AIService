"""
Model candidates per agent — V12 Multi-Model Support.

Each agent has an ordered list of OpenRouter model IDs to try.
SafeLLM will attempt them in order, falling back to Groq if all fail.

Models are ordered by capability (strongest first):
- openai/gpt-oss-120b:free  — strongest reasoning, best for complex tasks
- z-ai/glm-4.5-air:free     — strong multilingual, good for synthesis
- openai/gpt-oss-20b:free   — lighter, faster, good backup
"""

MODEL_CANDIDATES: dict[str, list[str]] = {
    "planner": [
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
    ],
    "clarifier": [
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b:free",
    ],
    "researcher": [
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
    ],
    "writer": [
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
    ],
    "reviewer": [
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
    ],
    "fast_chat": [
        "openai/gpt-oss-20b:free",
        "openai/gpt-oss-120b:free",
    ],
}
