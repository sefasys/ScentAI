from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = "scentai-hf-publisher"
MODEL_VOLUME_NAME = "scentai-models"
HF_SECRET_NAME = "scentai-huggingface"
ADAPTER_DIR = Path("/models/scentai")
BASE_MODEL_NAME = "google/gemma-4-12B-it"
DEFAULT_REPO_NAME = "scentai"
EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
KNOWN_PILOT_WEIGHT_HASHES = {
    "2919cc970fc4961e9a0a4cfa3fe2612c7e2b68aba625f5f5ba96c640e36237cd",
    "7b3224f19feb194ab339409de532d704b13cb693cc1995738ef5ea2f6e0b59b8",
    "2c66c65fd82080f3883408447cddf50cbaff98b48b2b87b4269bb2ad74be0bc3",
}

ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD = ROOT / "model" / "README.md"
MODEL_LICENSE = ROOT / "model" / "LICENSE.md"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

publisher_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface-hub==0.36.2",
        "numpy==2.2.6",
        "safetensors==0.6.2",
    )
    .add_local_file(MODEL_CARD, "/assets/README.md", copy=True)
    .add_local_file(MODEL_LICENSE, "/assets/LICENSE.md", copy=True)
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adapter(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Final adapter must contain adapter_config.json and adapter_model.safetensors: {adapter_dir}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights_sha256 = sha256_file(weights_path)
    problems = []
    if config.get("base_model_name_or_path") != BASE_MODEL_NAME:
        problems.append("base model does not match the frozen Gemma 4 release")
    if config.get("peft_type") != "LORA":
        problems.append("adapter is not LoRA")
    if int(config.get("r") or 0) != 16:
        problems.append("adapter rank is not 16")
    if bool(config.get("use_dora")):
        problems.append("deployment adapter must not use DoRA")
    if set(config.get("target_modules") or []) != EXPECTED_TARGET_MODULES:
        problems.append("target modules do not match the evaluated release")
    if weights_path.stat().st_size < 100_000_000:
        problems.append("adapter weights are unexpectedly small")
    if weights_sha256 in KNOWN_PILOT_WEIGHT_HASHES:
        problems.append("adapter weights match a known pilot checkpoint")
    if problems:
        raise ValueError("Refusing to publish an unexpected adapter: " + "; ".join(problems))

    from safetensors import safe_open

    # NumPy is sufficient for structural inspection and keeps this CPU-only
    # publisher independent of the much larger PyTorch runtime.
    with safe_open(weights_path, framework="numpy") as handle:
        tensor_keys = list(handle.keys())
    if not tensor_keys or not all("lora_" in key for key in tensor_keys):
        raise ValueError("Adapter safetensors does not contain the expected LoRA tensors")

    return {
        "base_model": config["base_model_name_or_path"],
        "rank": int(config["r"]),
        "alpha": int(config["lora_alpha"]),
        "target_modules": sorted(config["target_modules"]),
        "tensor_count": len(tensor_keys),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": weights_sha256,
        "config_sha256": sha256_file(config_path),
    }


@app.function(
    image=publisher_image,
    secrets=[hf_secret],
    volumes={"/models": model_volume},
    cpu=2.0,
    memory=4096,
    timeout=1800,
)
def publish_adapter(repo_name: str = DEFAULT_REPO_NAME) -> dict[str, Any]:
    from huggingface_hub import HfApi

    clean_repo_name = repo_name.strip()
    if not clean_repo_name or "/" in clean_repo_name:
        raise ValueError("repo_name must be an unqualified Hugging Face repository name")

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    identity = api.whoami()
    owner = str(identity.get("name") or "").strip()
    if not owner:
        raise RuntimeError("Could not determine the Hugging Face account from HF_TOKEN")
    repo_id = f"{owner}/{clean_repo_name}"
    adapter_report = validate_adapter(ADAPTER_DIR)

    with tempfile.TemporaryDirectory(prefix="scentai-hf-") as temporary:
        release_dir = Path(temporary)
        shutil.copy2(ADAPTER_DIR / "adapter_config.json", release_dir / "adapter_config.json")
        shutil.copy2(ADAPTER_DIR / "adapter_model.safetensors", release_dir / "adapter_model.safetensors")
        shutil.copy2("/assets/README.md", release_dir / "README.md")
        shutil.copy2("/assets/LICENSE.md", release_dir / "LICENSE.md")
        manifest = {
            "schema_version": 1,
            "artifact": "ScentAI Gemma 4 12B LoRA",
            "repo_id": repo_id,
            "adapter": adapter_report,
            "training_dataset": (
                "https://www.kaggle.com/datasets/sefasoysal/"
                "scentai-32k-grounded-perfume-conversations"
            ),
        }
        (release_dir / "adapter_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=release_dir,
            commit_message="Publish the evaluated ScentAI LoRA adapter",
        )

    return {
        "status": "ok",
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "private": True,
        "commit": str(commit),
        "adapter": adapter_report,
    }


@app.local_entrypoint()
def main(repo_name: str = DEFAULT_REPO_NAME) -> None:
    print(json.dumps(publish_adapter.remote(repo_name), indent=2, ensure_ascii=False))
