from __future__ import annotations

import json
import os
import sys
from pathlib import Path


EXPECTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    value = float(os.environ.get(name, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def validate_adapter(adapter_dir: Path, model_name: str, max_rank: int) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"LoRA mount must contain adapter_config.json and adapter_model.safetensors: {adapter_dir}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    problems = []
    if config.get("base_model_name_or_path") != model_name:
        problems.append("base_model_name_or_path does not match SCENTAI_BASE_MODEL")
    rank = int(config.get("r") or 0)
    if rank <= 0 or rank > max_rank:
        problems.append(f"adapter rank {rank} is outside vLLM max rank {max_rank}")
    if config.get("use_dora", False):
        problems.append("DoRA adapters are not supported by this frozen vLLM deployment")
    targets = set(config.get("target_modules") or [])
    if targets != EXPECTED_TARGETS:
        problems.append(f"target_modules mismatch: {sorted(targets)}")
    if problems:
        raise ValueError("Invalid ScentAI adapter: " + "; ".join(problems))
    return config


def main() -> None:
    model_name = os.environ.get("SCENTAI_BASE_MODEL", "google/gemma-4-12B-it").strip()
    lora_name = os.environ.get("SCENTAI_LORA_NAME", "scentai").strip()
    adapter_dir = Path(os.environ.get("SCENTAI_ADAPTER_DIR", "/models/scentai"))
    max_model_len = env_int("SCENTAI_MAX_MODEL_LEN", 4096, minimum=512)
    max_lora_rank = env_int("SCENTAI_MAX_LORA_RANK", 16)
    gpu_memory = env_float(
        "SCENTAI_GPU_MEMORY_UTILIZATION",
        0.65,
        minimum=0.25,
        maximum=0.98,
    )
    validate_adapter(adapter_dir, model_name, max_lora_rank)

    descriptor = json.dumps(
        {"name": lora_name, "path": str(adapter_dir), "base_model_name": model_name},
        separators=(",", ":"),
    )
    args = [
        "vllm",
        "serve",
        model_name,
        "--host",
        "0.0.0.0",
        "--port",
        "8010",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory),
        "--enable-prefix-caching",
        "--enable-lora",
        "--lora-modules",
        descriptor,
        "--max-lora-rank",
        str(max_lora_rank),
        "--max-loras",
        "1",
        "--generation-config",
        "vllm",
    ]
    print(json.dumps({"event": "starting_vllm", "command": args}), flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"event": "startup_error", "error": str(exc)}), file=sys.stderr, flush=True)
        raise
