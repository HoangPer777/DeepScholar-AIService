#!/usr/bin/env python3
"""Check which OpenRouter free models are currently usable.

Usage:
    python scripts/check_openrouter_free_models.py
    python scripts/check_openrouter_free_models.py --limit 30 --timeout 20
    python scripts/check_openrouter_free_models.py --output openrouter_free_models_report.json

The script reads OPENROUTER_API_KEY from the environment or from .env in the
current project directory. It lists OpenRouter models, filters free models, then
sends a tiny chat completion request to each model to detect which ones are
usable right now.
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

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_PROMPT = "Reply with exactly: ok"


@dataclass
class ModelCheckResult:
    model_id: str
    ok: bool
    status_code: int | None
    error_type: str | None
    latency_seconds: float | None
    output_sample: str | None
    context_length: int | None
    prompt_price: str | None
    completion_price: str | None


def load_api_key() -> str:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    load_dotenv()
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "DeepScholar OpenRouter Free Model Check",
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


def price_is_free(value: Any) -> bool:
    if value is None:
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return str(value).strip() in {"0", "0.0", "$0", "$0.0"}


def is_free_model(model: dict[str, Any]) -> bool:
    model_id = str(model.get("id", ""))
    if model_id.endswith(":free"):
        return True

    pricing = model.get("pricing") or {}
    prompt_price = pricing.get("prompt")
    completion_price = pricing.get("completion")
    return price_is_free(prompt_price) and price_is_free(completion_price)


def fetch_free_models(api_key: str, timeout: float) -> list[dict[str, Any]]:
    response = requests.get(
        OPENROUTER_MODELS_URL,
        headers=auth_headers(api_key),
        timeout=timeout,
    )
    if response.status_code != 200:
        error_type = classify_error(response.status_code, response.text)
        raise RuntimeError(
            f"Cannot list OpenRouter models: HTTP {response.status_code} ({error_type})\n"
            f"{response.text[:500]}"
        )

    payload = response.json()
    models = payload.get("data", [])
    if not isinstance(models, list):
        raise RuntimeError("Unexpected OpenRouter /models response: 'data' is not a list")

    free_models = [model for model in models if is_free_model(model)]
    free_models.sort(
        key=lambda model: (
            not str(model.get("id", "")).endswith(":free"),
            str(model.get("id", "")),
        )
    )
    return free_models


def check_model(api_key: str, model: dict[str, Any], timeout: float) -> ModelCheckResult:
    model_id = str(model.get("id", ""))
    pricing = model.get("pricing") or {}
    context_length = model.get("context_length")

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": DEFAULT_PROMPT}],
        "temperature": 0,
        "max_tokens": 8,
    }

    started = time.perf_counter()
    try:
        response = requests.post(
            OPENROUTER_CHAT_URL,
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
            context_length=context_length if isinstance(context_length, int) else None,
            prompt_price=str(pricing.get("prompt")) if pricing.get("prompt") is not None else None,
            completion_price=str(pricing.get("completion")) if pricing.get("completion") is not None else None,
        )
    except requests.RequestException as exc:
        return ModelCheckResult(
            model_id=model_id,
            ok=False,
            status_code=None,
            error_type=f"network_error: {type(exc).__name__}",
            latency_seconds=None,
            output_sample=None,
            context_length=context_length if isinstance(context_length, int) else None,
            prompt_price=str(pricing.get("prompt")) if pricing.get("prompt") is not None else None,
            completion_price=str(pricing.get("completion")) if pricing.get("completion") is not None else None,
        )

    if response.status_code != 200:
        return ModelCheckResult(
            model_id=model_id,
            ok=False,
            status_code=response.status_code,
            error_type=classify_error(response.status_code, response.text),
            latency_seconds=latency,
            output_sample=response.text[:160].replace("\n", " "),
            context_length=context_length if isinstance(context_length, int) else None,
            prompt_price=str(pricing.get("prompt")) if pricing.get("prompt") is not None else None,
            completion_price=str(pricing.get("completion")) if pricing.get("completion") is not None else None,
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
        context_length=context_length if isinstance(context_length, int) else None,
        prompt_price=str(pricing.get("prompt")) if pricing.get("prompt") is not None else None,
        completion_price=str(pricing.get("completion")) if pricing.get("completion") is not None else None,
    )


def print_results(results: list[ModelCheckResult]) -> None:
    usable = [result for result in results if result.ok]
    failed = [result for result in results if not result.ok]

    print("\nAvailable OpenRouter free models:")
    if usable:
        for index, result in enumerate(usable, start=1):
            context = f", ctx={result.context_length}" if result.context_length else ""
            print(f"{index:>2}. {result.model_id:<55} OK  {result.latency_seconds}s{context}")
    else:
        print("No usable free model found in this run.")

    if failed:
        print("\nFree models that failed the test:")
        for result in failed:
            status = result.status_code if result.status_code is not None else "-"
            print(f"- {result.model_id:<55} HTTP {status}  {result.error_type}")

    if usable:
        recommended = ",".join(result.model_id for result in usable)
        fast = ",".join(result.model_id for result in usable[: min(5, len(usable))])
        print("\nRecommended .env:")
        print(f"OPENROUTER_PLANNER_MODELS={recommended}")
        print(f"OPENROUTER_CLARIFIER_MODELS={fast}")
        print(f"OPENROUTER_RESEARCHER_MODELS={recommended}")
        print(f"OPENROUTER_WRITER_MODELS={recommended}")
        print(f"OPENROUTER_REVIEWER_MODELS={recommended}")
        print(f"OPENROUTER_FAST_CHAT_MODELS={fast}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find OpenRouter free models that are currently usable."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of free models to test after listing models. Default: 50.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds for each request. Default: 20.",
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
            "Missing OPENROUTER_API_KEY. Add it to your environment or DeepScholar-AIService/.env.",
            file=sys.stderr,
        )
        return 2

    print("Listing OpenRouter free models...")
    try:
        free_models = fetch_free_models(api_key, args.timeout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.limit > 0:
        free_models = free_models[: args.limit]

    print(f"Found {len(free_models)} free models to test.")
    results: list[ModelCheckResult] = []
    for index, model in enumerate(free_models, start=1):
        model_id = model.get("id", "<unknown>")
        print(f"[{index}/{len(free_models)}] Testing {model_id} ...", flush=True)
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
