import random
import time
import urllib.request


def fetch_with_retry(url: str, headers: dict | None = None, timeout: int = 12, retries: int = 3) -> bytes:
    err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            err = exc
            backoff = (2 ** i) + random.uniform(0, 0.4)
            time.sleep(backoff)
    raise RuntimeError(f"HTTP fetch failed after retries: {err}")
