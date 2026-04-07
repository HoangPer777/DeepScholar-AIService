from app.core.utils import effective_question, log
from app.tools.vector_search import search_article_chunks
from app.workflows.states import AgentState


class ReaderAgent:
    """RAG agent — queries PGVector for relevant article chunks. No LLM needed."""

    def run(self, state: AgentState) -> AgentState:
        if not state.article_id:
            log(state, "\n[ReaderAgent] No article_id — skipping vector search")
            return state

        try:
            chunks = search_article_chunks(
                article_id=state.article_id,
                question=effective_question(state),
                focus_sections=state.focus_sections,
                limit=8,
            )
            state.vector_context = [
                {
                    "content":  c["content"],
                    "chunk_id": c["chunk_id"],
                    "distance": c["distance"],
                    "section":  "unknown",
                }
                for c in chunks
            ]
            log(state, f"\n[ReaderAgent] Retrieved {len(chunks)} chunks from PGVector")
        except Exception as e:
            log(state, f"\n[ReaderAgent] WARNING: vector search failed ({e}) — continuing without PDF context")
            state.vector_context = []
        return state
