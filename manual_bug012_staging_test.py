"""Read-only interactive smoke test for BUG-012 on an isolated staging service.

This script only POSTs to Backend auth/login to obtain short-lived access tokens.
All AIService requests are GETs; it does not create chat/research jobs or call LLMs.
Run from a terminal so passwords can be entered through getpass without echo.
"""

from __future__ import annotations

import getpass
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


TIMEOUT_SECONDS = 12
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _normalize_base_url(value: str, label: str) -> str:
    value = value.strip().rstrip("/")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be a complete http(s) URL.")
    if parsed.scheme != "https" and parsed.hostname not in LOCAL_HOSTS:
        raise ValueError(f"{label} must use HTTPS unless it is localhost.")
    return value


def _request_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        # Do not print response bodies; they may contain private user data.
        return exc.code, None

    if not raw:
        return status, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, None


def _login(backend_base: str, label: str) -> tuple[str, str]:
    email = input(f"{label} QA email: ").strip()
    password = getpass.getpass(f"{label} QA password (hidden): ")
    if not email or not password:
        raise RuntimeError(f"{label} email/password cannot be empty.")

    status, response = _request_json(
        "POST",
        f"{backend_base}/auth/login/",
        payload={"email": email, "password": password},
    )
    password = ""
    if status != 200 or not isinstance(response, dict) or not response.get("access"):
        raise RuntimeError(f"{label} login failed with HTTP {status}; stopping without further requests.")
    return email.casefold(), str(response["access"])


def _assert_status(label: str, actual: int, expected: set[int]) -> bool:
    if actual in expected:
        print(f"[PASS] {label}: HTTP {actual}")
        return True
    print(f"[FAIL] {label}: HTTP {actual}; expected {' or '.join(map(str, sorted(expected)))}")
    return False


def _get_sessions(ai_base: str, token: str) -> dict[str, Any]:
    status, data = _request_json("GET", f"{ai_base}/chat/sessions", token=token)
    if status != 200 or not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
        raise RuntimeError(f"Authenticated session-list request failed with HTTP {status}.")
    return data


def main() -> int:
    print("BUG-012 staging read-only check")
    print("Only login POSTs and AIService GETs are sent. No chat/research is created.\n")

    backend_base = _normalize_base_url(input("Staging Backend base (include /api/v1): "), "Backend URL")
    ai_base = _normalize_base_url(input("Staging AIService base (include /api): "), "AIService URL")
    if input("Type STAGING-ONLY to confirm both URLs are isolated staging, not production: ").strip() != "STAGING-ONLY":
        print("Stopped before making requests.")
        return 2

    session_id = input("Existing QA session_id owned by account A: ").strip()
    try:
        parsed_session = uuid.UUID(session_id)
    except (ValueError, AttributeError):
        print("Invalid session UUID; stopped before login.")
        return 2
    session_id = str(parsed_session)

    token_a = ""
    token_b = ""
    try:
        email_a, token_a = _login(backend_base, "A")
        email_b, token_b = _login(backend_base, "B")
        if email_a == email_b:
            raise RuntimeError("A and B must be two distinct QA accounts; stopping.")

        all_passed = True

        # This request must be rejected before session data is read.
        anonymous_status, _ = _request_json("GET", f"{ai_base}/chat/sessions")
        all_passed &= _assert_status("Unauthenticated session list", anonymous_status, {401})

        sessions_a = _get_sessions(ai_base, token_a)
        ids_a = {str(item.get("session_id")) for item in sessions_a["sessions"] if isinstance(item, dict)}
        if session_id not in ids_a:
            raise RuntimeError("Session A is not listed for account A; refusing cross-user probe.")

        status_a, _ = _request_json(
            "GET", f"{ai_base}/chat/history/{session_id}", token=token_a
        )
        all_passed &= _assert_status("A reads A history", status_a, {200})

        sessions_b = _get_sessions(ai_base, token_b)
        ids_b = {str(item.get("session_id")) for item in sessions_b["sessions"] if isinstance(item, dict)}
        if session_id in ids_b:
            all_passed = False
            print("[FAIL] A's session appears in B's session list.")
        else:
            print("[PASS] A's session is absent from B's session list.")

        status_b, _ = _request_json(
            "GET", f"{ai_base}/chat/history/{session_id}", token=token_b
        )
        all_passed &= _assert_status("B reads A history", status_b, {404})

        print("\nNo response bodies, passwords, or tokens were printed.")
        return 0 if all_passed else 1
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        print(f"STOPPED: {exc}")
        return 2
    finally:
        # Best-effort reference cleanup; access tokens were never written to disk.
        token_a = ""
        token_b = ""


if __name__ == "__main__":
    sys.exit(main())
