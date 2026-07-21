from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HOST = "127.0.0.1"
PORT = 18020
BASE_URL = f"http://{HOST}:{PORT}"


def request_json(path: str, payload: dict | None = None, *, timeout: int = 30):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> None:
    environment = os.environ.copy()
    environment.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SCENTAI_CHROMA_DIR": str(ROOT / "chroma_db_bge_m3"),
        "SCENTAI_CATALOG_PATH": str(ROOT / "scentai_catalog.sqlite3"),
        "SCENTAI_RETRIEVAL_MAX_CONCURRENCY": "1",
        "PYTHONPATH": os.pathsep.join([str(ROOT / "deploy/src"), str(ROOT / "src"), str(ROOT)]),
    })
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "scentai_deploy.retrieval_api:app",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"Retrieval server exited during startup:\n{output}")
            try:
                status, health = request_json("/health", timeout=2)
                if status == 200:
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(0.25)
        else:
            raise TimeoutError("Retrieval HTTP service did not become ready within 45 seconds")

        _, resolved = request_json("/resolve", {"hint": "Club de Nuit by Armaf"})
        _, search = request_json(
            "/search",
            {"query": "clean office fragrance", "top_k": 3, "exclude_terms": ["vanilla"]},
        )
        assert health["collection_count"] == 131_930
        assert resolved["resolved"]["name"] == "Club de Nuit Intense Man"
        assert len(search["results"]) == 3
        print(json.dumps({
            "status": "ok",
            "health": health["status"],
            "resolved": resolved["resolved"]["label"],
            "results": [item["label"] for item in search["results"]],
        }, indent=2, ensure_ascii=False))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
