from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scentai.orchestrator import RetrievalClient

from .config import ApiSettings
from .runtime import RuntimeBundle


EXPECTED_ADAPTER_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class ModalJsonClient:
    """Expose a Modal class method through the tiny JsonHttpClient contract."""

    def __init__(self, worker: Any) -> None:
        self.worker = worker

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path, None)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        response = self.worker.request.remote(method, path, payload)
        if not isinstance(response, dict):
            raise RuntimeError(f"Modal worker returned a non-object response for {path}")
        status_code = int(response.get("status_code") or 500)
        body = response.get("body")
        if status_code >= 400:
            raise RuntimeError(f"Modal worker returned {status_code} from {path}: {body}")
        if not isinstance(body, dict):
            raise RuntimeError(f"Modal worker returned a non-JSON body for {path}")
        return body


def build_modal_runtime(
    settings: ApiSettings,
    model_worker: Any,
    retrieval_worker: Any,
) -> RuntimeBundle:
    model_http = ModalJsonClient(model_worker)
    retrieval_client = RetrievalClient(ModalJsonClient(retrieval_worker))
    return RuntimeBundle.create_with_clients(settings, model_http, retrieval_client)


def dispatch_retrieval(
    engine: Any,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_method = method.upper()
    payload = payload or {}
    if normalized_method == "GET" and path in {"/health", "/health/live", "/health/ready"}:
        body = engine.health()
        return {"status_code": 200 if body.get("status") == "ok" else 503, "body": body}
    operations = {
        ("POST", "/search"): engine.search,
        ("POST", "/resolve"): engine.resolve,
        ("POST", "/similar"): engine.similar,
    }
    operation = operations.get((normalized_method, path))
    if operation is None:
        return {"status_code": 404, "body": {"detail": "Unknown retrieval operation"}}
    try:
        return {"status_code": 200, "body": operation(payload)}
    except LookupError as exc:
        return {"status_code": 404, "body": {"detail": str(exc)}}
    except (TypeError, ValueError) as exc:
        return {"status_code": 400, "body": {"detail": str(exc)}}


def forward_json_request(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method.upper(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {"status_code": int(response.status), "body": body}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"detail": raw}
        return {"status_code": int(exc.code), "body": body}


def wait_for_json_endpoint(
    url: str,
    *,
    timeout_seconds: int,
    interval_seconds: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(interval_seconds)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def validate_modal_artifacts(
    model_root: Path,
    data_root: Path,
    *,
    base_model_name: str = "google/gemma-4-12B-it",
    max_lora_rank: int = 16,
) -> dict[str, Any]:
    adapter_dir = model_root / "scentai"
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    chroma_path = data_root / "chroma_db_bge_m3" / "chroma.sqlite3"
    catalog_path = data_root / "scentai_catalog.sqlite3"
    missing = [
        str(path)
        for path in (config_path, weights_path, chroma_path, catalog_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing Modal artifacts: " + ", ".join(missing))

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if adapter_config.get("base_model_name_or_path") != base_model_name:
        problems.append("base_model_name_or_path does not match the frozen Gemma 4 model")
    rank = int(adapter_config.get("r") or 0)
    if rank <= 0 or rank > max_lora_rank:
        problems.append(f"adapter rank {rank} is outside max rank {max_lora_rank}")
    if adapter_config.get("use_dora", False):
        problems.append("the release adapter must not use DoRA")
    targets = set(adapter_config.get("target_modules") or [])
    if targets != EXPECTED_ADAPTER_TARGETS:
        problems.append(f"target_modules mismatch: {sorted(targets)}")
    if problems:
        raise ValueError("Invalid Modal adapter: " + "; ".join(problems))

    return {
        "status": "ok",
        "base_model": base_model_name,
        "adapter": {
            "path": str(adapter_dir),
            "rank": rank,
            "weights_bytes": weights_path.stat().st_size,
        },
        "retrieval": {
            "chroma_path": str(chroma_path),
            "chroma_bytes": chroma_path.stat().st_size,
            "catalog_path": str(catalog_path),
            "catalog_bytes": catalog_path.stat().st_size,
        },
    }
