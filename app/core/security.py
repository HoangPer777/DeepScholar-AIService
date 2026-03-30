import hmac
from typing import Optional

from fastapi import Header, HTTPException
from jose import jwt, JWTError

from core.config import get_settings


def sanitize_question(question: str) -> str:
    blocked = ["```", "<|", "|>", "[[", "]]", "SYSTEM:", "ASSISTANT:", "TOOL:"]
    clean = question
    for token in blocked:
        clean = clean.replace(token, "")
    clean = " ".join(clean.split())
    return clean.strip()


def verify_service_key(x_api_key: Optional[str]) -> None:
    settings = get_settings()
    if not settings.auth_required:
        return
    if not settings.service_api_key:
        raise HTTPException(status_code=500, detail="SERVICE_API_KEY is missing")
    if not x_api_key or not hmac.compare_digest(x_api_key, settings.service_api_key):
        raise HTTPException(status_code=401, detail="Invalid service API key")


def verify_bearer_token(authorization: Optional[str]) -> dict:
    settings = get_settings()
    if not settings.auth_required:
        return {"sub": "anonymous", "scope": "research:readwrite"}

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.replace("Bearer ", "", 1).strip()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_public_key,
            algorithms=["RS256"],
            issuer=settings.jwt_issuer or None,
            audience=settings.jwt_audience or None,
        )
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    return payload


def guard_prompt_injection(text: str) -> None:
    suspicious = [
        "ignore previous instructions",
        "reveal system prompt",
        "you are now developer",
        "execute tool",
        "sudo",
    ]
    lowered = text.lower()
    if any(token in lowered for token in suspicious):
        raise HTTPException(status_code=400, detail="Prompt appears malicious or policy-unsafe")
