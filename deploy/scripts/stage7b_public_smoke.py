from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 ScentAI-Stage7B-Smoke/1.0",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def poll_job(
    url: str,
    *,
    deadline_seconds: int,
    initial_poll_after_ms: int,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    deadline = started + deadline_seconds
    poll_after_ms = initial_poll_after_ms
    previous_status = None
    last_progress_at = 0.0

    while time.monotonic() < deadline:
        time.sleep(max(0.25, min(poll_after_ms / 1000, 5.0)))
        status, body = request_json(url)
        if status != 200:
            raise RuntimeError(f"Unexpected job status HTTP {status}: {body}")
        job_status = body.get("status")
        elapsed = time.monotonic() - started
        if job_status != previous_status or elapsed - last_progress_at >= 30:
            print(f"{url.rsplit('/', 1)[-1]}: {job_status} ({elapsed:.1f}s)", flush=True)
            previous_status = job_status
            last_progress_at = elapsed
        if job_status == "succeeded":
            return body, elapsed
        if job_status == "failed":
            raise RuntimeError(body.get("error") or f"Job failed: {body}")
        poll_after_ms = int(body.get("poll_after_ms") or poll_after_ms)

    raise TimeoutError(f"Job did not finish within {deadline_seconds} seconds: {url}")


def run_public_smoke(base_url: str, *, stage: str = "stage7_public_gateway") -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    api_url = base_url + "/api/scentai"
    started = time.monotonic()

    live_status, live = request_json(api_url + "/health/live")
    if live_status != 200 or live.get("status") != "ok":
        raise RuntimeError(f"Public health proxy failed: HTTP {live_status} {live}")

    warmup_status, warmup = request_json(
        api_url + "/v1/runtime/warmup/jobs",
        method="POST",
    )
    if warmup_status not in {200, 202} or not warmup.get("job_id"):
        raise RuntimeError(f"Warm-up job was not accepted: HTTP {warmup_status} {warmup}")
    warmup_job, warmup_seconds = poll_job(
        api_url + f"/v1/runtime/warmup/jobs/{warmup['job_id']}",
        deadline_seconds=15 * 60,
        initial_poll_after_ms=int(warmup.get("poll_after_ms") or 2_000),
    )
    if not warmup_job.get("ready"):
        raise RuntimeError(f"Warm-up completed without ready=true: {warmup_job}")

    chat_status, chat = request_json(
        api_url + "/v1/chat/jobs",
        method="POST",
        payload={"query": "YSL Y EDP nasıl bir parfüm?"},
    )
    if chat_status not in {200, 202} or not chat.get("job_id"):
        raise RuntimeError(f"Chat job was not accepted: HTTP {chat_status} {chat}")
    chat_job, chat_seconds = poll_job(
        api_url + f"/v1/chat/jobs/{chat['job_id']}",
        deadline_seconds=12 * 60,
        initial_poll_after_ms=int(chat.get("poll_after_ms") or 1_500),
    )
    response = chat_job.get("response") or {}
    labels = [item.get("label") for item in response.get("recommendations") or []]
    failures = []
    if not response.get("validation_passed"):
        failures.append("validation_failed")
    if not str(response.get("answer") or "").strip():
        failures.append("empty_answer")
    if "Y Eau de Parfum by Yves Saint Laurent" not in labels:
        failures.append("entity_resolution_mismatch")

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "url": base_url,
        "pass": not failures,
        "failures": failures,
        "warmup_seconds": round(warmup_seconds, 4),
        "chat_seconds": round(chat_seconds, 4),
        "total_seconds": round(time.monotonic() - started, 4),
        "warmup_job": warmup_job,
        "chat_response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the public Stage 7B Cloudflare gateway")
    parser.add_argument("--url", required=True, help="Public Worker URL without a trailing slash")
    parser.add_argument("--stage", default="stage7_public_gateway", help="Stage label stored in the report")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("deploy/reports/stage7b_public_smoke.json"),
    )
    args = parser.parse_args()

    report = run_public_smoke(args.url, stage=args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", args.output)
    if not report["pass"]:
        raise SystemExit(f"Stage 7B smoke failed: {report['failures']}")
    print("STAGE 7B PUBLIC SMOKE: PASSED")


if __name__ == "__main__":
    main()
