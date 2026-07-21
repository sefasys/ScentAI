from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal


APP_NAME = "scentai-stage7-job-smoke"
API_SECRET_NAME = "scentai-api"
DEFAULT_URL = "https://m-sefa-soysal--scentai-stage6-web.modal.run"

app = modal.App(APP_NAME)
image = modal.Image.debian_slim(python_version="3.12")
api_secret = modal.Secret.from_name(API_SECRET_NAME, required_keys=["SCENTAI_API_KEY"])


def _request_json(
    url: str,
    api_key: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


@app.function(image=image, secrets=[api_secret], timeout=900)
def run_job_smoke(base_url: str, query: str) -> dict[str, Any]:
    base_url = base_url.rstrip("/")
    api_key = os.environ["SCENTAI_API_KEY"]
    started = time.monotonic()
    warmup_accepted = _request_json(
        base_url + "/v1/runtime/warmup/jobs",
        api_key,
        method="POST",
    )
    warmup_job_id = str(warmup_accepted["job_id"])
    warmup_polls = 0
    last_warmup_status = None
    while time.monotonic() - started < 840:
        time.sleep(max(5.0, float(warmup_accepted.get("poll_after_ms", 2000)) / 1000))
        warmup_polls += 1
        warmup = _request_json(
            base_url + f"/v1/runtime/warmup/jobs/{warmup_job_id}",
            api_key,
        )
        if warmup["status"] != last_warmup_status or warmup_polls % 12 == 0:
            print(json.dumps({"warmup_poll": warmup_polls, "status": warmup["status"]}), flush=True)
            last_warmup_status = warmup["status"]
        if warmup["status"] == "succeeded" and warmup.get("ready"):
            break
        if warmup["status"] == "failed":
            return {
                "pass": False,
                "warmup": {
                    "job_id": warmup_job_id,
                    "polls": warmup_polls,
                    "status": warmup["status"],
                    "error": warmup.get("error"),
                },
                "elapsed_seconds": round(time.monotonic() - started, 4),
            }
    else:
        raise TimeoutError("Warm-up smoke exceeded 840 seconds")

    warmup_elapsed = round(time.monotonic() - started, 4)
    accepted = _request_json(
        base_url + "/v1/chat/jobs",
        api_key,
        method="POST",
        payload={"query": query},
    )
    job_id = str(accepted["job_id"])
    polls = 0
    last_status = None
    while time.monotonic() - started < 840:
        time.sleep(max(5.0, float(accepted.get("poll_after_ms", 1500)) / 1000))
        polls += 1
        job = _request_json(base_url + f"/v1/chat/jobs/{job_id}", api_key)
        if job["status"] != last_status or polls % 12 == 0:
            print(json.dumps({"poll": polls, "status": job["status"]}), flush=True)
            last_status = job["status"]
        if job["status"] == "succeeded":
            response = job["response"]
            return {
                "pass": bool(response.get("validation_passed")),
                "warmup": {
                    "job_id": warmup_job_id,
                    "polls": warmup_polls,
                    "status": "succeeded",
                    "elapsed_seconds": warmup_elapsed,
                },
                "job_id": job_id,
                "polls": polls,
                "elapsed_seconds": round(time.monotonic() - started, 4),
                "response": response,
            }
        if job["status"] == "failed":
            return {
                "pass": False,
                "job_id": job_id,
                "polls": polls,
                "elapsed_seconds": round(time.monotonic() - started, 4),
                "error": job.get("error"),
                "error_status": job.get("error_status"),
            }
    raise TimeoutError("Job smoke exceeded 840 seconds")


@app.local_entrypoint()
def main(
    url: str = DEFAULT_URL,
    query: str = "Ofis için temiz, profesyonel ve vanilyasız üç parfüm öner.",
    output: str = "deploy/reports/modal_job_http_smoke.json",
) -> None:
    report = run_job_smoke.remote(url, query)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("pass", "polls", "elapsed_seconds")}, indent=2))
    print("Saved:", output_path)
    if not report["pass"]:
        raise SystemExit(1)
