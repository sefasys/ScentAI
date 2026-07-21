from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable


CASES = (
    {
        "name": "clean_office",
        "query": "I need exactly 3 clean professional office fragrances without vanilla.",
        "language": "en",
        "count": 3,
    },
    {
        "name": "entity_resolution",
        "query": "YSL Y EDP nasıl bir parfüm?",
        "language": "tr",
        "contains": "Y Eau de Parfum by Yves Saint Laurent",
    },
    {
        "name": "comparison",
        "query": "Compare Club de Nuit by Armaf with Team Five by Adidas for vibe and use.",
        "language": "en",
        "count": 2,
    },
    {
        "name": "unsupported_price",
        "query": "What is the current price of Aventus by Creed?",
        "language": "en",
        "route": "unsupported_price",
        "count": 0,
    },
)


def request_json(
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def score_response(case: dict[str, Any], body: dict[str, Any]) -> list[str]:
    failures = []
    if not body.get("validation_passed"):
        failures.append("validation_failed")
    if body.get("language") != case["language"]:
        failures.append("language_mismatch")
    if "route" in case and body.get("route") != case["route"]:
        failures.append("route_mismatch")
    if "count" in case and len(body.get("recommendations") or []) != case["count"]:
        failures.append("recommendation_count_mismatch")
    labels = {item.get("label") for item in body.get("recommendations") or []}
    if "contains" in case and case["contains"] not in labels:
        failures.append("entity_resolution_mismatch")
    return failures


def run_http_smoke(
    base_url: str,
    api_key: str,
    *,
    requester: Callable[..., dict[str, Any]] = request_json,
) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    live = requester(base_url + "/health/live")
    if live.get("status") != "ok":
        raise RuntimeError(f"Liveness failed: {live}")

    outputs = []
    for case in CASES:
        started = time.perf_counter()
        body = requester(
            base_url + "/v1/chat",
            api_key=api_key,
            payload={"query": case["query"]},
        )
        failures = score_response(case, body)
        outputs.append(
            {
                "name": case["name"],
                "pass": not failures,
                "failures": failures,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "response": body,
            }
        )
        print(case["name"], "PASS" if not failures else f"FAIL {failures}", flush=True)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "url": base_url,
        "pass": all(output["pass"] for output in outputs),
        "outputs": outputs,
    }
