import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict


request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("deep_research")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_request_id(request_id: str | None = None) -> str:
    rid = request_id or str(uuid.uuid4())
    request_id_ctx.set(rid)
    return rid