from dataclasses import dataclass

import httpx
from fastapi import Header, HTTPException

from app.core.config import settings


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str


async def get_current_user(authorization: str | None = Header(default=None)) -> AuthenticatedUser:
    """Resolve the bearer token through Backend; never trust client-supplied identity."""
    scheme, separator, token = (authorization or "").partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    url = f"{settings.BACKEND_API_URL.rstrip('/')}/auth/me/"
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            response = await client.get(url, headers={"Authorization": f"Bearer {token.strip()}"})
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail="Authentication service unavailable.") from exc

    if response.status_code in {401, 403}:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not response.is_success:
        raise HTTPException(status_code=503, detail="Authentication service unavailable.")

    try:
        user = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Authentication service returned an invalid response.") from exc

    if not isinstance(user, dict):
        raise HTTPException(status_code=503, detail="Authentication service returned an invalid response.")
    user_id = user.get("id")
    if isinstance(user_id, bool) or not isinstance(user_id, (int, str)) or not str(user_id).strip():
        raise HTTPException(status_code=503, detail="Authentication service returned an invalid identity.")
    if user.get("is_active") is not True:
        raise HTTPException(status_code=401, detail="User account is inactive.", headers={"WWW-Authenticate": "Bearer"})

    return AuthenticatedUser(user_id=str(user_id))
