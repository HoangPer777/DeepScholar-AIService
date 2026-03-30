import time
import uuid
from typing import List, Dict, Any, Optional

import psycopg

from core.config import get_settings


def _conn() -> psycopg.Connection:
    settings = get_settings()
    if not settings.postgres_dsn:
        raise RuntimeError("DATABASE_URL is required for persistence")
    dsn = settings.postgres_dsn
    if dsn.startswith("postgresql+psycopg://"):
        dsn = dsn.replace("postgresql+psycopg://", "postgresql://", 1)
    return psycopg.connect(dsn)


def create_session(user_id: str, question: str) -> str:
    session_id = str(uuid.uuid4())
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_sessions (id, user_id, question, created_at)
            VALUES (%s, %s, %s, %s)
            """,
            (session_id, user_id, question, int(time.time())),
        )
        conn.commit()
    return session_id


def save_session_result(session_id: str, answer: str, confidence: float, logs: List[Dict[str, Any]]) -> None:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE research_sessions
            SET answer=%s, confidence_score=%s, logs_json=%s, finished_at=%s
            WHERE id=%s
            """,
            (answer, confidence, logs, int(time.time()), session_id),
        )
        conn.commit()


def append_message(session_id: str, role: str, content: str) -> None:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_messages (session_id, role, content, created_at)
            VALUES (%s, %s, %s, %s)
            """,
            (session_id, role, content, int(time.time())),
        )
        conn.commit()


def get_session(session_id: str) -> Optional[dict]:
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, question, answer, confidence_score, logs_json, created_at, finished_at
            FROM research_sessions WHERE id=%s
            """,
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "user_id": row[1],
            "question": row[2],
            "answer": row[3],
            "confidence_score": float(row[4] or 0.0),
            "logs": row[5] or [],
            "created_at": row[6],
            "finished_at": row[7],
        }
