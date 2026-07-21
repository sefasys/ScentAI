from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("/content/drive/MyDrive/Perfume-Dataset")
DEFAULT_ADAPTER_RELATIVE = Path(
    "models/scentai-gemma-4-12b-it-full-fastmodel-lora/best_lora_adapter"
)


def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args[:4]), "..." if len(args) > 4 else "", flush=True)
    return subprocess.run(args, check=check, text=True)


def ensure_modal() -> None:
    try:
        import modal  # noqa: F401
    except ImportError:
        command(sys.executable, "-m", "pip", "install", "modal==1.5.2")


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def ensure_volume(name: str) -> None:
    result = subprocess.run(
        ["modal", "volume", "create", name],
        check=False,
        text=True,
        capture_output=True,
    )
    message = (result.stdout or "") + (result.stderr or "")
    if result.returncode and "already exists" not in message.lower():
        raise RuntimeError(f"Could not create volume {name}: {message.strip()}")
    print(f"Volume ready: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ScentAI artifacts from Drive to Modal")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    ensure_modal()

    if not os.environ.get("MODAL_TOKEN_ID"):
        os.environ["MODAL_TOKEN_ID"] = getpass.getpass("Modal token ID: ").strip()
    if not os.environ.get("MODAL_TOKEN_SECRET"):
        os.environ["MODAL_TOKEN_SECRET"] = getpass.getpass("Modal token secret: ").strip()
    if not os.environ["MODAL_TOKEN_ID"] or not os.environ["MODAL_TOKEN_SECRET"]:
        raise ValueError("Both Modal token values are required")

    root = args.project_root
    adapter = args.adapter or root / DEFAULT_ADAPTER_RELATIVE
    chroma = require_path(root / "chroma_db_bge_m3", "Chroma directory")
    catalog = require_path(root / "scentai_catalog.sqlite3", "catalog")
    adapter = require_path(adapter, "full LoRA adapter")
    require_path(adapter / "adapter_config.json", "adapter_config.json")
    require_path(adapter / "adapter_model.safetensors", "adapter weights")

    for volume in ("scentai-models", "scentai-data"):
        ensure_volume(volume)

    command("modal", "volume", "put", "--force", "scentai-models", str(adapter), "/scentai")
    command("modal", "volume", "put", "--force", "scentai-data", str(chroma), "/chroma_db_bge_m3")
    command("modal", "volume", "put", "--force", "scentai-data", str(catalog), "/scentai_catalog.sqlite3")
    command("modal", "volume", "ls", "scentai-models", "/scentai")
    command("modal", "volume", "ls", "scentai-data", "/")
    print("\nArtifact upload complete. Run the Modal artifact preflight next.")


if __name__ == "__main__":
    main()
