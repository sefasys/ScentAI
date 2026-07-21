# %% [markdown]
# # ScentAI Clean Inference Bootstrap
#
# This notebook starts from zero and validates only one boundary:
#
# `Gemma 4 12B base model + the existing ScentAI LoRA -> vLLM HTTP response`
#
# It intentionally does **not** install or import Chroma, BGE, sentence-transformers,
# SciPy, sklearn, torchvision, Unsloth, TRL, or PEFT. Those belong to later stages.
# vLLM and its own PyTorch/CUDA dependencies live in a clean uv environment, never
# in the Colab notebook kernel.
#
# Use a fresh A100 runtime and choose **Run all**. Do not run this after another
# notebook has modified the session.

# %%
import shutil
import subprocess
import sys


def run_checked(command, *, label):
    print(f"\n[{label}]", " ".join(map(str, command)))
    subprocess.run([str(part) for part in command], check=True)


# uv is a standalone environment manager. --no-deps guarantees that installing
# this launcher cannot replace NumPy, Torch, Transformers, or any Colab library.
run_checked(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "uv"],
    label="install uv launcher",
)
UV = shutil.which("uv")
assert UV, "uv was installed but its executable is not on PATH."
print(subprocess.check_output([UV, "--version"], text=True).strip())

# %% [markdown]
# ## Build one isolated inference environment
#
# vLLM chooses and owns its compatible PyTorch build. No other ML package is added
# to this environment, and nothing from it is imported into the notebook kernel.

# %%
from pathlib import Path

INFERENCE_ENV = Path("/content/scentai_inference_env")
INFERENCE_PYTHON = INFERENCE_ENV / "bin" / "python"
VLLM_EXECUTABLE = INFERENCE_ENV / "bin" / "vllm"
NINJA_EXECUTABLE = INFERENCE_ENV / "bin" / "ninja"
VLLM_VERSION = "0.25.1"

run_checked(
    [
        UV,
        "venv",
        "--clear",
        "--python",
        "3.12",
        "--seed",
        "--managed-python",
        str(INFERENCE_ENV),
    ],
    label="create clean Python 3.12 environment",
)
run_checked(
    [
        UV,
        "pip",
        "install",
        "--python",
        str(INFERENCE_PYTHON),
        f"vllm=={VLLM_VERSION}",
        "ninja",
        "--torch-backend=auto",
        "--strict",
    ],
    label="install vLLM with its matching PyTorch backend",
)
run_checked(
    [UV, "pip", "check", "--python", str(INFERENCE_PYTHON)],
    label="check isolated dependency graph",
)

assert INFERENCE_PYTHON.exists(), INFERENCE_PYTHON
assert VLLM_EXECUTABLE.exists(), VLLM_EXECUTABLE
assert NINJA_EXECUTABLE.exists(), NINJA_EXECUTABLE
print("ninja:", subprocess.check_output([str(NINJA_EXECUTABLE), "--version"], text=True).strip())

# %% [markdown]
# ## Binary and GPU smoke test
#
# This test runs in a subprocess. A failure here stops immediately, before Drive is
# mounted or 24 GB of model weights are downloaded.

# %%
import json

probe_code = r"""
import json
from importlib.metadata import version
import torch
import vllm

payload = {
    "vllm_distribution": version("vllm"),
    "vllm_module": vllm.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print("SCENTAI_PROBE=" + json.dumps(payload))
"""
probe = subprocess.run(
    [str(INFERENCE_PYTHON), "-c", probe_code],
    text=True,
    capture_output=True,
)
if probe.returncode:
    print(probe.stdout)
    print(probe.stderr)
    raise RuntimeError("The isolated vLLM binary smoke test failed.")

probe_line = next(
    (line for line in probe.stdout.splitlines() if line.startswith("SCENTAI_PROBE=")),
    None,
)
assert probe_line, probe.stdout
inference_versions = json.loads(probe_line.split("=", 1)[1])
print(json.dumps(inference_versions, indent=2))
assert inference_versions["vllm_distribution"].startswith(VLLM_VERSION), inference_versions
assert inference_versions["cuda_available"], "Enable an A100 GPU runtime in Colab."

nvidia_smi = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ],
    text=True,
    capture_output=True,
)
if nvidia_smi.returncode:
    raise RuntimeError("nvidia-smi failed:\n" + nvidia_smi.stderr)
gpu_row = nvidia_smi.stdout.strip().splitlines()[0]
gpu_name, gpu_memory_mib = [part.strip() for part in gpu_row.rsplit(",", 1)]
gpu_memory_gb = float(gpu_memory_mib) / 1024
print(f"GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
assert gpu_memory_gb >= 35, "Use an A100-class runtime with at least 35 GB of GPU memory."

# %% [markdown]
# ## Mount Drive and validate the adapter

# %%
from google.colab import drive, userdata

drive.mount("/content/drive")

PROJECT_DIR = Path("/content/drive/MyDrive/Perfume-Dataset")
ADAPTER_DIR = (
    PROJECT_DIR
    / "models"
    / "scentai-gemma-4-12b-it-pilot-fastmodel-lora"
    / "best_lora_adapter"
)
MODEL_NAME = "google/gemma-4-12B-it"
LORA_NAME = "scentai"

adapter_config_path = ADAPTER_DIR / "adapter_config.json"
adapter_weights_path = ADAPTER_DIR / "adapter_model.safetensors"
assert adapter_config_path.exists(), f"Missing adapter config: {adapter_config_path}"
assert adapter_weights_path.exists(), f"Missing adapter weights: {adapter_weights_path}"

adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
assert adapter_config.get("base_model_name_or_path") == MODEL_NAME, adapter_config
assert int(adapter_config.get("r") or 0) == 16, adapter_config
assert not adapter_config.get("use_dora", False), "The current vLLM path expects the standard LoRA."
language_lora_targets = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
adapter_targets = set(adapter_config.get("target_modules") or [])
assert adapter_targets == language_lora_targets, adapter_targets
print(
    "Adapter preflight:",
    {
        "base": adapter_config["base_model_name_or_path"],
        "rank": adapter_config["r"],
        "alpha": adapter_config.get("lora_alpha"),
        "language_only_targets": sorted(adapter_targets),
        "weights_mb": round(adapter_weights_path.stat().st_size / (1024 ** 2), 1),
    },
)

try:
    HF_TOKEN = userdata.get("HF_TOKEN") or ""
except Exception:
    HF_TOKEN = ""
print("HF token:", "available" if HF_TOKEN else "not set; the public base repository will be used")

# %% [markdown]
# ## Start the local vLLM server
#
# Both the base model and the LoRA are served by one process. The notebook kernel
# communicates with it through localhost and never imports vLLM.

# %%
import os
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"
PORT = 8010
BASE_URL = f"http://{HOST}:{PORT}"
SERVER_LOG = Path("/content/scentai_vllm_server.log")
START_TIMEOUT_SECONDS = 2400


def get_json(url, *, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def server_is_healthy():
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def log_tail(line_count=80):
    if not SERVER_LOG.exists():
        return "<server log has not been created>"
    lines = SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


lora_descriptor = json.dumps(
    {
        "name": LORA_NAME,
        "path": str(ADAPTER_DIR),
        "base_model_name": MODEL_NAME,
    }
)
server_command = [
    str(VLLM_EXECUTABLE),
    "serve",
    MODEL_NAME,
    "--host",
    HOST,
    "--port",
    str(PORT),
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "4096",
    "--gpu-memory-utilization",
    "0.86",
    "--enable-prefix-caching",
    "--enable-lora",
    "--lora-modules",
    lora_descriptor,
    "--max-lora-rank",
    "16",
    "--max-loras",
    "1",
    "--generation-config",
    "vllm",
]

required_models = {MODEL_NAME, LORA_NAME}
if server_is_healthy():
    model_ids = {item["id"] for item in get_json(f"{BASE_URL}/v1/models").get("data", [])}
    if not required_models.issubset(model_ids):
        raise RuntimeError(
            f"Port {PORT} belongs to another server. Found {sorted(model_ids)}. "
            "Restart the Colab runtime and Run all."
        )
    print("Reusing healthy ScentAI vLLM server.")
else:
    server_environment = os.environ.copy()
    server_environment["PYTHONUNBUFFERED"] = "1"
    server_environment["HF_HOME"] = "/content/huggingface_cache"
    # The vLLM entry point uses the isolated interpreter, but child JIT processes
    # search PATH for build tools. Explicitly expose the environment's ninja.
    server_environment["PATH"] = (
        str(INFERENCE_ENV / "bin")
        + os.pathsep
        + server_environment.get("PATH", "")
    )
    if HF_TOKEN:
        server_environment["HF_TOKEN"] = HF_TOKEN

    server_log_handle = SERVER_LOG.open("w", encoding="utf-8")
    server_process = subprocess.Popen(
        server_command,
        stdout=server_log_handle,
        stderr=subprocess.STDOUT,
        env=server_environment,
    )
    print("Starting vLLM. The first base-model download and compile can take several minutes.")
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    next_status = time.monotonic() + 30
    while not server_is_healthy():
        return_code = server_process.poll()
        if return_code is not None:
            server_log_handle.close()
            raise RuntimeError(
                f"vLLM exited with code {return_code}. Last log lines:\n{log_tail()}"
            )
        if time.monotonic() >= deadline:
            server_process.terminate()
            server_log_handle.close()
            raise TimeoutError(
                f"vLLM was not healthy after {START_TIMEOUT_SECONDS}s. Last log lines:\n{log_tail()}"
            )
        if time.monotonic() >= next_status:
            print("Still loading...\n" + log_tail(10))
            next_status = time.monotonic() + 30
        time.sleep(3)

    model_ids = {item["id"] for item in get_json(f"{BASE_URL}/v1/models").get("data", [])}
    if not required_models.issubset(model_ids):
        server_process.terminate()
        server_log_handle.close()
        raise RuntimeError(
            f"Server started without both model IDs. Found {sorted(model_ids)}.\n{log_tail()}"
        )

print("Served models:", sorted(model_ids))
print("vLLM server preflight: passed")

# %% [markdown]
# ## API smoke tests
#
# The first request proves the base model endpoint. The second selects the LoRA by
# model name and proves that the existing adapter can generate a grounded answer.

# %%
def post_json(path, payload, *, timeout=600):
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {body}") from exc


def chat(model, messages, *, max_tokens=160):
    response = post_json(
        "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM returned no choices: {response}")
    content = str(choices[0].get("message", {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(f"vLLM returned an empty answer: {response}")
    return content, response.get("usage") or {}


base_answer, base_usage = chat(
    MODEL_NAME,
    [{"role": "user", "content": "Reply with exactly: BASE_MODEL_OK"}],
    max_tokens=16,
)
print("Base response:", base_answer)
print("Base usage:", base_usage)

lora_smoke_prompt = """You are ScentAI, a grounded perfume consultant.
Use only the supplied perfume cards and do not invent facts.

[PERFUMES]
Versace Pour Homme by Versace
Accords: floral, musky, fresh spicy, citrus, aromatic, green, fresh
Best seasons: spring, summer
Time: day

Prada L'Homme by Prada
Accords: iris, powdery, clean, woody, amber
Best seasons: spring, summer, autumn
Time: day

Question: Recommend one clean office scent without vanilla and briefly explain why."""
lora_answer, lora_usage = chat(
    LORA_NAME,
    [{"role": "user", "content": lora_smoke_prompt}],
    max_tokens=180,
)
print("\nLoRA response:\n" + lora_answer)
print("LoRA usage:", lora_usage)
assert any(name in lora_answer for name in ("Versace Pour Homme", "Prada L'Homme")), (
    "The LoRA answered, but did not mention a supplied perfume."
)
print("\nCLEAN INFERENCE SMOKE TEST: PASSED")

# %% [markdown]
# ## Interactive inference-only test
#
# This final cell deliberately has no retrieval. Supply the context yourself while
# validating model behavior. Retrieval will be added later as a separate service,
# after this notebook passes unchanged.

# %%
my_prompt = """You are ScentAI. Use only the supplied perfume card.

[PERFUMES]
Aventus Cologne by Creed
Accords: musky, leather, citrus, fresh spicy, aromatic, smoky, woody, powdery
Best seasons: spring, summer
Time: day

Question: Describe this fragrance's likely character and best wear context."""

answer, usage = chat(
    LORA_NAME,
    [{"role": "user", "content": my_prompt}],
    max_tokens=220,
)
print(answer)
print("Usage:", usage)
