#!/usr/bin/env python3
"""Check which Groq models are currently usable and measure their latency.

Usage:
    python scripts/check_groq_free_models.py
    python scripts/check_groq_free_models.py --timeout 15
    python scripts/check_groq_free_models.py --output groq_free_models_report.json

The script reads GROQ_API_KEY from the environment or from .env in the
current project directory. It lists Groq models, filters text/chat models, then
sends a tiny chat completion request to each model to detect which ones are
usable right now and measure their response times.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_PROMPT = "Reply with exactly: ok"


@dataclass
class ModelCheckResult:
    model_id: str
    ok: bool
    status_code: int | None
    error_type: str | None
    latency_seconds: float | None
    output_sample: str | None
    context_window: int | None
    owned_by: str | None


def load_api_key() -> str:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    load_dotenv()
    return os.getenv("GROQ_API_KEY", "").strip()


def auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def classify_error(status_code: int | None, body: str) -> str:
    lowered = body.lower()
    if status_code == 429:
        return "rate_limited"
    if status_code in {401, 403}:
        return "auth_error"
    if status_code == 404 or "not found" in lowered:
        return "model_not_found"
    if status_code in {408, 504} or "timeout" in lowered:
        return "timeout"
    if "context" in lowered and "length" in lowered:
        return "context_error"
    if status_code is not None and 500 <= status_code <= 599:
        return "provider_unavailable"
    return "unknown"


def is_chat_model(model: dict[str, Any]) -> bool:
    model_id = str(model.get("id", "")).lower()
    # Exclude audio (whisper), embedding, vision/multimodal if they aren't compatible with chat, guardrails etc.
    exclude_keywords = {"whisper", "embed", "guard", "tts", "audio", "lc-whisper"}
    if any(keyword in model_id for keyword in exclude_keywords):
        return False
    return True


def fetch_groq_models(api_key: str, timeout: float) -> list[dict[str, Any]]:
    response = requests.get(
        GROQ_MODELS_URL,
        headers=auth_headers(api_key),
        timeout=timeout,
    )
    if response.status_code != 200:
        error_type = classify_error(response.status_code, response.text)
        raise RuntimeError(
            f"Cannot list Groq models: HTTP {response.status_code} ({error_type})\n"
            f"{response.text[:500]}"
        )

    payload = response.json()
    models = payload.get("data", [])
    if not isinstance(models, list):
        raise RuntimeError("Unexpected Groq /models response: 'data' is not a list")

    chat_models = [model for model in models if is_chat_model(model)]
    chat_models.sort(key=lambda m: str(m.get("id", "")))
    return chat_models


def check_model(api_key: str, model: dict[str, Any], timeout: float) -> ModelCheckResult:
    model_id = str(model.get("id", ""))
    context_window = model.get("context_window")
    owned_by = model.get("owned_by")

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
        "temperature": 0,
        "max_tokens": 8,
    }

    started = time.perf_counter()
    try:
        response = requests.post(
            GROQ_CHAT_URL,
            headers=auth_headers(api_key),
            json=payload,
            timeout=timeout,
        )
        latency = round(time.perf_counter() - started, 3)
    except requests.Timeout:
        return ModelCheckResult(
            model_id=model_id,
            ok=False,
            status_code=None,
            error_type="timeout",
            latency_seconds=None,
            output_sample=None,
            context_window=context_window if isinstance(context_window, int) else None,
            owned_by=owned_by,
        )
    except requests.RequestException as exc:
        return ModelCheckResult(
            model_id=model_id,
            ok=False,
            status_code=None,
            error_type=f"network_error: {type(exc).__name__}",
            latency_seconds=None,
            output_sample=None,
            context_window=context_window if isinstance(context_window, int) else None,
            owned_by=owned_by,
        )

    if response.status_code != 200:
        return ModelCheckResult(
            model_id=model_id,
            ok=False,
            status_code=response.status_code,
            error_type=classify_error(response.status_code, response.text),
            latency_seconds=latency,
            output_sample=response.text[:160].replace("\n", " "),
            context_window=context_window if isinstance(context_window, int) else None,
            owned_by=owned_by,
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"].get("content", "")
    except (KeyError, IndexError, TypeError, ValueError):
        content = response.text[:160]

    return ModelCheckResult(
        model_id=model_id,
        ok=True,
        status_code=response.status_code,
        error_type=None,
        latency_seconds=latency,
        output_sample=str(content).strip()[:160],
        context_window=context_window if isinstance(context_window, int) else None,
        owned_by=owned_by,
    )


def print_results(results: list[ModelCheckResult]) -> None:
    usable = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]

    print("\nAvailable Groq models:")
    if usable:
        for index, result in enumerate(usable, start=1):
            context = f", ctx={result.context_window}" if result.context_window else ""
            print(f"{index:>2}. {result.model_id:<55} OK  {result.latency_seconds}s{context}")
    else:
        print("No usable model found in this run.")

    if failed:
        print("\nModels that failed the test:")
        for result in failed:
            status = result.status_code if result.status_code is not None else "-"
            print(f"- {result.model_id:<55} HTTP {status}  {result.error_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find Groq models that are currently usable."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds for each request. Default: 15.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = load_api_key()
    if not api_key:
        print(
            "Missing GROQ_API_KEY. Add it to your environment or DeepScholar-AIService/.env.",
            file=sys.stderr,
        )
        return 2

    print("Listing Groq models...")
    try:
        models = fetch_groq_models(api_key, args.timeout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Found {len(models)} chat models to test.")
    results: list[ModelCheckResult] = []
    for index, model in enumerate(models, start=1):
        model_id = model.get("id", "<unknown>")
        print(f"[{index}/{len(models)}] Testing {model_id} ...", flush=True)
        result = check_model(api_key, model, args.timeout)
        results.append(result)

    results.sort(key=lambda result: (not result.ok, result.latency_seconds or 999999, result.model_id))
    print_results(results)

    if args.output:
        report = {
            "checked_at_unix": int(time.time()),
            "total_tested": len(results),
            "usable_count": sum(1 for result in results if result.ok),
            "results": [asdict(result) for result in results],
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {args.output}")

    return 0 if any(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
