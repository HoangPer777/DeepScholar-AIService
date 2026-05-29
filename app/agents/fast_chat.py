from langchain_core.messages import HumanMessage, SystemMessage

from app.prompts.fast_chat_prompt import FAST_CHAT_SYSTEM_PROMPT
from app.schemas.chat_models import ContextWindow


class FastChatAgent:
    """Single-LLM-call agent for follow-up questions when Research_Context is available.

    Makes exactly one ``self.llm.invoke()`` call per ``run()`` invocation.
    Builds context from the research report answer, up to 20 sources, and the
    last 10 conversation messages (Requirements 3.3, 3.4).
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, question: str, context: ContextWindow) -> dict:
        """Answer a follow-up question using the cached Research_Context.

        Args:
            question: The follow-up question from the user.
            context:  The session's ContextWindow containing the research report,
                      sources, and conversation history.

        Returns:
            dict with keys:
                answer            – LLM-generated conversational answer (str)
                citations         – list of source dicts (up to 20)
                confidence_score  – float in [0, 1]
                need_clarification – bool
                is_fast_chat      – True (always)
        """
        report = context.research_report

        # Build numbered source list — cap at 20 (Requirement 3.4)
        sources_text = "\n".join(
            f"[{s.index}] {s.title} ({s.url})"
            for s in context.sources[:20]
        )

        # Last 10 messages from conversation history (Requirement 3.4)
        history_text = "\n".join(
            f"{m.role.upper()}: {m.content}"
            for m in context.messages[-10:]
        )

        user_content = (
            f"RESEARCH REPORT:\n{report.answer if report else 'N/A'}\n\n"
            f"SOURCES:\n{sources_text}\n\n"
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"FOLLOW-UP QUESTION: {question}"
        )

        # Exactly one LLM call (Requirement 3.3)
        response = self.llm.invoke([
            SystemMessage(content=FAST_CHAT_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])

        return {
            "answer": response.content,
            "citations": [s.model_dump() for s in context.sources[:20]],
            "confidence_score": 0.85,  # fast chat is high-confidence by design
            "need_clarification": False,
            "is_fast_chat": True,
        }
