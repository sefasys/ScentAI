from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import modal


APP_NAME = "scentai-stage6"
BASE_MODEL_NAME = "google/gemma-4-12B-it"
LORA_MODEL_NAME = "scentai"
MODEL_VOLUME_NAME = "scentai-models"
DATA_VOLUME_NAME = "scentai-data"
HF_CACHE_VOLUME_NAME = "scentai-hf-cache"
VLLM_CACHE_VOLUME_NAME = "scentai-vllm-cache"
HF_SECRET_NAME = "scentai-huggingface"
API_SECRET_NAME = "scentai-api"
MODEL_IMAGE_TAG = "vllm/vllm-openai:v0.25.1-x86_64-cu129"
MODEL_GPU = "A100-80GB"

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = ROOT / "deploy"

app = modal.App(APP_NAME)

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name(VLLM_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
api_secret = modal.Secret.from_name(API_SECRET_NAME, required_keys=["SCENTAI_API_KEY"])


def _add_runtime_source(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir(ROOT / "src", "/app/src", copy=True)
        .add_local_dir(DEPLOYMENT_DIR / "src", "/app/deploy/src", copy=True)
        .env({"PYTHONPATH": "/app/src:/app/deploy/src", "PYTHONUNBUFFERED": "1"})
    )


api_image = _add_runtime_source(
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "fastapi==0.139.0",
        "pydantic==2.13.4",
    )
).add_local_file(
    ROOT / "evaluation" / "final_eval_v1.jsonl",
    "/app/evaluation/final_eval_v1.jsonl",
    copy=True,
)

retrieval_image = _add_runtime_source(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install(
        "torch==2.11.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "chromadb==1.5.9",
        "fastapi==0.139.0",
        "pydantic==2.13.4",
        "sentence-transformers==5.6.0",
    )
)

model_image = _add_runtime_source(
    modal.Image.from_registry(MODEL_IMAGE_TAG, add_python="3.12")
    .entrypoint([])
    .run_commands(
        "set -eux; command -v vllm; head -n 1 $(command -v vllm); "
        "/usr/bin/python3 -c \"import vllm; print(vllm.__version__)\""
    )
    .add_local_file(
        DEPLOYMENT_DIR / "model_server" / "entrypoint.py",
        "/opt/scentai/entrypoint.py",
        copy=True,
    )
)


@app.cls(
    image=model_image,
    gpu=MODEL_GPU,
    secrets=[hf_secret],
    volumes={
        "/models": model_volume,
        "/cache/huggingface": hf_cache_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    env={
        "SCENTAI_BASE_MODEL": BASE_MODEL_NAME,
        "SCENTAI_LORA_NAME": LORA_MODEL_NAME,
        "SCENTAI_ADAPTER_DIR": "/models/scentai",
        "SCENTAI_MAX_MODEL_LEN": "4096",
        "SCENTAI_MAX_LORA_RANK": "16",
        "SCENTAI_GPU_MEMORY_UTILIZATION": "0.65",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "TOKENIZERS_PARALLELISM": "false",
    },
    min_containers=0,
    max_containers=1,
    scaledown_window=300,
    timeout=600,
    startup_timeout=1800,
)
@modal.concurrent(max_inputs=1)
class ModelWorker:
    def _terminate_process(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)

    @modal.enter()
    def start(self) -> None:
        self.process = subprocess.Popen(
            ["python3", "/opt/scentai/entrypoint.py"],
            env=os.environ.copy(),
            start_new_session=True,
        )
        try:
            from scentai_deploy.modal_bridge import wait_for_json_endpoint

            models = wait_for_json_endpoint(
                "http://127.0.0.1:8010/v1/models",
                timeout_seconds=1500,
                interval_seconds=3.0,
            )
            served = {str(item.get("id")) for item in models.get("data", [])}
            required = {BASE_MODEL_NAME, LORA_MODEL_NAME}
            if not required.issubset(served):
                raise RuntimeError(f"vLLM started without required models: {sorted(served)}")
            vllm_cache_volume.commit()
        except Exception:
            self._terminate_process()
            raise

    @modal.method()
    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from scentai_deploy.modal_bridge import forward_json_request

        return forward_json_request(
            "http://127.0.0.1:8010",
            method,
            path,
            payload,
            timeout_seconds=540,
        )

    @modal.exit()
    def stop(self) -> None:
        self._terminate_process()


@app.cls(
    image=retrieval_image,
    volumes={
        "/data": data_volume,
        "/cache/huggingface": hf_cache_volume,
    },
    env={
        "SCENTAI_CHROMA_DIR": "/data/chroma_db_bge_m3",
        "SCENTAI_CATALOG_PATH": "/data/scentai_catalog.sqlite3",
        "SCENTAI_RETRIEVAL_MAX_CONCURRENCY": "1",
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
    },
    cpu=4.0,
    memory=16384,
    min_containers=0,
    max_containers=1,
    scaledown_window=1200,
    timeout=180,
    startup_timeout=900,
)
@modal.concurrent(max_inputs=1)
class RetrievalWorker:
    @modal.enter()
    def start(self) -> None:
        from scentai_deploy.config import RetrievalSettings
        from scentai_deploy.retrieval_api import RetrievalRuntime

        self.runtime = RetrievalRuntime(RetrievalSettings.from_env())

    @modal.method()
    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from scentai_deploy.modal_bridge import dispatch_retrieval

        future = self.runtime.executor.submit(
            dispatch_retrieval,
            self.runtime.engine,
            method,
            path,
            payload,
        )
        return future.result()

    @modal.exit()
    def stop(self) -> None:
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            runtime.close()


@app.function(
    image=api_image,
    secrets=[api_secret],
    env={
        "SCENTAI_BASE_MODEL": BASE_MODEL_NAME,
        "SCENTAI_LORA_NAME": LORA_MODEL_NAME,
        "SCENTAI_MODEL_TIMEOUT_SECONDS": "540",
        "SCENTAI_RETRIEVAL_TIMEOUT_SECONDS": "180",
        "SCENTAI_SESSION_TTL_SECONDS": "3600",
        "SCENTAI_MAX_SESSIONS": "1000",
        "SCENTAI_REQUEST_WORKER_THREADS": "4",
        "SCENTAI_EXPOSE_DEBUG": "false",
        "SCENTAI_CORS_ORIGINS": "http://127.0.0.1:5173,http://localhost:5173",
    },
    cpu=2.0,
    memory=2048,
    min_containers=0,
    max_containers=1,
    scaledown_window=1200,
    timeout=600,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    from scentai_deploy.api import app as api
    from scentai_deploy.modal_bridge import build_modal_runtime

    model_worker = ModelWorker()
    retrieval_worker = RetrievalWorker()
    api.state.runtime_factory = lambda settings: build_modal_runtime(
        settings,
        model_worker,
        retrieval_worker,
    )
    return api


@app.function(
    image=api_image,
    env={
        "SCENTAI_BASE_MODEL": BASE_MODEL_NAME,
        "SCENTAI_LORA_NAME": LORA_MODEL_NAME,
        "SCENTAI_MODEL_TIMEOUT_SECONDS": "540",
        "SCENTAI_RETRIEVAL_TIMEOUT_SECONDS": "180",
        "SCENTAI_SESSION_TTL_SECONDS": "7200",
        "SCENTAI_MAX_SESSIONS": "100",
        "SCENTAI_REQUEST_WORKER_THREADS": "1",
    },
    cpu=1.0,
    memory=2048,
    max_containers=1,
    timeout=7200,
)
def run_cloud_regression(limit: int = 12) -> dict[str, Any]:
    from scentai_deploy.config import ApiSettings
    from scentai_deploy.modal_bridge import build_modal_runtime
    from scentai_deploy.modal_regression import run_regression

    runtime = build_modal_runtime(ApiSettings.from_env(), ModelWorker(), RetrievalWorker())
    try:
        return run_regression(
            runtime,
            Path("/app/evaluation/final_eval_v1.jsonl"),
            limit=None if limit <= 0 else limit,
        )
    finally:
        runtime.close()


@app.function(
    image=api_image,
    secrets=[api_secret],
    cpu=1.0,
    memory=1024,
    timeout=3600,
)
def run_public_api_smoke(url: str) -> dict[str, Any]:
    from scentai_deploy.http_smoke import run_http_smoke

    return run_http_smoke(url, os.environ["SCENTAI_API_KEY"])


@app.local_entrypoint()
def regression(limit: int = 12, output: str = "deploy/reports/modal_regression.json") -> None:
    report = run_cloud_regression.remote(limit)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("case_count", "pass_count", "failure_count", "pass_rate", "fallback_rate", "elapsed_seconds")},
            indent=2,
        )
    )
    print("Saved:", output_path)
    if report["failure_count"]:
        raise SystemExit(1)


@app.local_entrypoint()
def public_smoke(
    url: str = "https://m-sefa-soysal--scentai-stage6-web.modal.run",
    output: str = "deploy/reports/modal_http_smoke.json",
) -> None:
    report = run_public_api_smoke.remote(url)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "case_count": len(report["outputs"])}, indent=2))
    print("Saved:", output_path)
    if not report["pass"]:
        raise SystemExit(1)
