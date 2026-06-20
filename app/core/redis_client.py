from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import redis as redis_lib

from app.core.config import settings

def mask_redis_url(redis_url: str) -> str:
    """Return a log-safe Redis URL without credentials."""
    try:
        parts = urlsplit(redis_url)
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        username = f"{parts.username}:***@" if parts.username else ""
        netloc = f"{username}{host}{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except Exception:
        return "<invalid redis url>"


def redis_connection_kwargs(decode_responses: bool = False) -> dict:
    return {
        "decode_responses": decode_responses,
        "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS,
        "socket_timeout": settings.REDIS_SOCKET_TIMEOUT_SECONDS,
        "health_check_interval": settings.REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
        "retry_on_timeout": True,
    }


def create_redis_client(
    *,
    decode_responses: bool = False,
    from_url: Optional[Callable] = None,
) -> redis_lib.Redis:
    """
    Create a Redis client from REDIS_URL.

    Supports both redis:// and rediss:// URLs. Use rediss:// for Upstash TLS.
    """
    factory = from_url or redis_lib.from_url
    return factory(settings.REDIS_URL, **redis_connection_kwargs(decode_responses))
