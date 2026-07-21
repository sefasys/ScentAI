from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal


APP_NAME = "scentai-stage6-setup"
BASE_MODEL_NAME = "google/gemma-4-12B-it"
MODEL_VOLUME_NAME = "scentai-models"
DATA_VOLUME_NAME = "scentai-data"
HF_CACHE_VOLUME_NAME = "scentai-hf-cache"
HF_SECRET_NAME = "scentai-huggingface"

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = ROOT / "deploy"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

setup_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface-hub==0.36.2",
        "hf-xet>=1.1.10,<2",
    )
    .add_local_dir(ROOT / "src", "/app/src", copy=True)
    .add_local_dir(DEPLOYMENT_DIR / "src", "/app/deploy/src", copy=True)
    .env({"PYTHONPATH": "/app/src:/app/deploy/src", "PYTHONUNBUFFERED": "1"})
)


@app.function(
    image=setup_image,
    volumes={"/models": model_volume, "/data": data_volume},
    cpu=1.0,
    memory=1024,
    timeout=120,
)
def artifact_preflight() -> dict[str, Any]:
    from scentai_deploy.modal_bridge import validate_modal_artifacts

    return validate_modal_artifacts(Path("/models"), Path("/data"))


@app.function(
    image=setup_image,
    secrets=[hf_secret],
    volumes={"/cache/huggingface": hf_cache_volume},
    cpu=4.0,
    memory=8192,
    timeout=7200,
)
def prefetch_huggingface() -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    cache_dir = "/cache/huggingface"
    downloaded = {}
    for repo_id in (BASE_MODEL_NAME, "BAAI/bge-m3"):
        downloaded[repo_id] = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=os.environ["HF_TOKEN"],
        )
    hf_cache_volume.commit()
    return {"status": "ok", "snapshots": downloaded}


@app.local_entrypoint()
def preflight() -> None:
    print(json.dumps(artifact_preflight.remote(), indent=2))


@app.local_entrypoint()
def prefetch() -> None:
    print(json.dumps(prefetch_huggingface.remote(), indent=2))
