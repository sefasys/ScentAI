from __future__ import annotations

import getpass
import json
import secrets
import subprocess
import tempfile
import argparse
from pathlib import Path


VOLUMES = (
    "scentai-models",
    "scentai-data",
    "scentai-hf-cache",
    "scentai-vllm-cache",
)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=not check)


def ensure_volume(name: str) -> None:
    result = run("modal", "volume", "create", name, check=False)
    if result.returncode and "already exists" not in (result.stdout + result.stderr).lower():
        raise RuntimeError(result.stderr or result.stdout)
    print(f"Volume ready: {name}")


def create_secret(name: str, values: dict[str, str]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
        Path(handle.name).chmod(0o600)
        json.dump(values, handle)
        handle.flush()
        run("modal", "secret", "create", "--force", "--from-json", handle.name, name)
    print(f"Secret ready: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ScentAI Modal volumes and secrets")
    parser.add_argument(
        "--cors-origin",
        default="",
        help="Optional frontend origin, for example https://scentai.example.com",
    )
    args = parser.parse_args()
    run("modal", "profile", "current")
    hf_token = getpass.getpass("Hugging Face token (input hidden): ").strip()
    if not hf_token:
        raise ValueError("HF token is required for the Gemma snapshot")
    api_key = getpass.getpass(
        "ScentAI public API key (blank generates a secure key; input hidden): "
    ).strip()
    generated = not api_key
    api_key = api_key or secrets.token_urlsafe(32)

    for name in VOLUMES:
        ensure_volume(name)
    create_secret("scentai-huggingface", {"HF_TOKEN": hf_token})
    api_values = {"SCENTAI_API_KEY": api_key}
    if args.cors_origin.strip():
        api_values["SCENTAI_CORS_ORIGINS"] = args.cors_origin.strip().rstrip("/")
    create_secret("scentai-api", api_values)

    print("\nModal Stage 6 bootstrap complete.")
    if generated:
        print("Generated SCENTAI_API_KEY (store it now):")
        print(api_key)


if __name__ == "__main__":
    main()
