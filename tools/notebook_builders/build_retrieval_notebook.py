from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2]
SERVICE_SOURCE = (ROOT / "src" / "scentai" / "retrieval.py").read_text(encoding="utf-8")
OUTPUT = ROOT / "notebooks" / "retrieval_colab.ipynb"


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


cells = [
    markdown(
        """
# ScentAI Clean Retrieval - Stage 2

This notebook validates only the retrieval boundary:

`structured request -> BGE-M3 + Chroma + community graph -> grounded candidates`

It does not install, import, start, or call vLLM. Retrieval runs in its own
CPU-only uv environment and exposes a localhost HTTP API. Use a fresh Colab
runtime and **Run all**. A GPU is not required for this stage.
"""
    ),
    code(
        """
import shutil
import subprocess
import sys


def run_checked(command, *, label):
    print(f"\\n[{label}]", " ".join(map(str, command)))
    subprocess.run([str(part) for part in command], check=True)


run_checked(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "uv"],
    label="install uv launcher",
)
UV = shutil.which("uv")
assert UV, "uv was installed but its executable is not on PATH."
print(subprocess.check_output([UV, "--version"], text=True).strip())
"""
    ),
    markdown(
        """
## Create a CPU-only retrieval environment

Only two top-level packages are requested. uv resolves their compatible NumPy,
SciPy, sklearn, Transformers, and CPU PyTorch versions inside this environment.
Nothing from this environment is imported into the notebook kernel.
"""
    ),
    code(
        """
from pathlib import Path

RETRIEVAL_ENV = Path("/content/scentai_retrieval_env")
RETRIEVAL_PYTHON = RETRIEVAL_ENV / "bin" / "python"

run_checked(
    [
        UV, "venv", "--clear", "--python", "3.12", "--seed", "--managed-python",
        str(RETRIEVAL_ENV),
    ],
    label="create clean retrieval environment",
)
run_checked(
    [
        UV, "pip", "install", "--python", str(RETRIEVAL_PYTHON),
        "chromadb==1.5.9", "sentence-transformers==5.6.0",
        "--torch-backend=cpu", "--strict",
    ],
    label="install retrieval dependencies with CPU PyTorch",
)
run_checked(
    [UV, "pip", "check", "--python", str(RETRIEVAL_PYTHON)],
    label="check retrieval dependency graph",
)
assert RETRIEVAL_PYTHON.exists(), RETRIEVAL_PYTHON
"""
    ),
    markdown("## Dependency smoke test"),
    code(
        """
import json

probe_code = r'''\
import json
from importlib.metadata import version
import chromadb
import numpy
import scipy
import sklearn
import sentence_transformers
import torch
import transformers

print("SCENTAI_RETRIEVAL_PROBE=" + json.dumps({
    "chromadb": version("chromadb"),
    "sentence_transformers": version("sentence-transformers"),
    "transformers": version("transformers"),
    "torch": version("torch"),
    "torch_cuda": torch.version.cuda,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
}))
'''
probe = subprocess.run(
    [str(RETRIEVAL_PYTHON), "-c", probe_code],
    text=True,
    capture_output=True,
)
if probe.returncode:
    print(probe.stdout)
    print(probe.stderr)
    raise RuntimeError("The isolated retrieval environment failed its import smoke test.")
probe_line = next(
    (line for line in probe.stdout.splitlines() if line.startswith("SCENTAI_RETRIEVAL_PROBE=")),
    None,
)
assert probe_line, probe.stdout
retrieval_versions = json.loads(probe_line.split("=", 1)[1])
print(json.dumps(retrieval_versions, indent=2))
assert retrieval_versions["chromadb"] == "1.5.9", retrieval_versions
assert retrieval_versions["sentence_transformers"] == "5.6.0", retrieval_versions
assert retrieval_versions["torch_cuda"] is None, (
    "Retrieval must use CPU-only PyTorch so it cannot reserve inference VRAM."
)
print("Retrieval dependency smoke test: passed")
"""
    ),
    markdown(
        """
## Mount Drive and stage immutable data locally

Reading the 1.5 GB HNSW index directly from Drive is unnecessarily slow. The
notebook copies Chroma and the 94 MB SQLite catalog to Colab's local disk once.
"""
    ),
    code(
        """
import os
import time
from google.colab import drive, userdata

drive.mount("/content/drive")
PROJECT_DIR = Path("/content/drive/MyDrive/Perfume-Dataset")
DRIVE_CHROMA_DIR = PROJECT_DIR / "chroma_db_bge_m3"
DRIVE_CATALOG = PROJECT_DIR / "scentai_catalog.sqlite3"
LOCAL_CHROMA_DIR = Path("/content/scentai_data/chroma_db_bge_m3")
LOCAL_CATALOG = Path("/content/scentai_data/scentai_catalog.sqlite3")

assert (DRIVE_CHROMA_DIR / "chroma.sqlite3").exists(), f"Missing Chroma DB: {DRIVE_CHROMA_DIR}"
assert DRIVE_CATALOG.exists(), f"Missing catalog: {DRIVE_CATALOG}"

copy_started = time.perf_counter()
if LOCAL_CHROMA_DIR.parent.exists():
    shutil.rmtree(LOCAL_CHROMA_DIR.parent)
LOCAL_CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(DRIVE_CHROMA_DIR, LOCAL_CHROMA_DIR)
shutil.copy2(DRIVE_CATALOG, LOCAL_CATALOG)
copy_elapsed = time.perf_counter() - copy_started

chroma_bytes = sum(path.stat().st_size for path in LOCAL_CHROMA_DIR.rglob("*") if path.is_file())
print(f"Local Chroma: {chroma_bytes / (1024 ** 3):.2f} GB")
print(f"Local catalog: {LOCAL_CATALOG.stat().st_size / (1024 ** 2):.1f} MB")
print(f"Local staging time: {copy_elapsed:.1f}s")
assert chroma_bytes > 1_000_000_000, "The copied Chroma index looks incomplete."
assert LOCAL_CATALOG.stat().st_size > 50_000_000, "The copied catalog looks incomplete."

try:
    HF_TOKEN = userdata.get("HF_TOKEN") or ""
except Exception:
    HF_TOKEN = ""
"""
    ),
    markdown("## Materialize and start the isolated retrieval service"),
    code(
        "SERVICE_SOURCE = " + repr(SERVICE_SOURCE) + "\n\n" + r'''
import urllib.error
import urllib.request

SERVICE_PATH = Path("/content/scentai_retrieval_service.py")
SERVICE_LOG = Path("/content/scentai_retrieval_service.log")
SERVICE_PATH.write_text(SERVICE_SOURCE, encoding="utf-8")
subprocess.run([str(RETRIEVAL_PYTHON), "-m", "py_compile", str(SERVICE_PATH)], check=True)

RETRIEVAL_HOST = "127.0.0.1"
RETRIEVAL_PORT = 8020
RETRIEVAL_URL = f"http://{RETRIEVAL_HOST}:{RETRIEVAL_PORT}"


def get_json(url, *, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def service_is_healthy():
    try:
        return get_json(f"{RETRIEVAL_URL}/health", timeout=2).get("status") == "ok"
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def service_log_tail(line_count=80):
    if not SERVICE_LOG.exists():
        return "<service log has not been created>"
    return "\\n".join(
        SERVICE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    )


if service_is_healthy():
    print("Reusing healthy retrieval service.")
else:
    service_environment = os.environ.copy()
    service_environment.update({
        "PYTHONUNBUFFERED": "1",
        "ANONYMIZED_TELEMETRY": "False",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HOME": "/content/huggingface_retrieval_cache",
        "SCENTAI_CHROMA_DIR": str(LOCAL_CHROMA_DIR),
        "SCENTAI_CATALOG_PATH": str(LOCAL_CATALOG),
        "SCENTAI_RETRIEVAL_HOST": RETRIEVAL_HOST,
        "SCENTAI_RETRIEVAL_PORT": str(RETRIEVAL_PORT),
        "OMP_NUM_THREADS": str(min(os.cpu_count() or 2, 8)),
    })
    if HF_TOKEN:
        service_environment["HF_TOKEN"] = HF_TOKEN
    service_log_handle = SERVICE_LOG.open("w", encoding="utf-8")
    service_process = subprocess.Popen(
        [str(RETRIEVAL_PYTHON), str(SERVICE_PATH)],
        stdout=service_log_handle,
        stderr=subprocess.STDOUT,
        env=service_environment,
    )
    print("Starting BGE-M3 retrieval service. First model download/load can take several minutes.")
    deadline = time.monotonic() + 1800
    next_status = time.monotonic() + 30
    while not service_is_healthy():
        return_code = service_process.poll()
        if return_code is not None:
            service_log_handle.close()
            raise RuntimeError(
                f"Retrieval service exited with code {return_code}. Last log lines:\\n{service_log_tail()}"
            )
        if time.monotonic() >= deadline:
            service_process.terminate()
            service_log_handle.close()
            raise TimeoutError("Retrieval service startup timed out.\\n" + service_log_tail())
        if time.monotonic() >= next_status:
            print("Still loading...\\n" + service_log_tail(10))
            next_status = time.monotonic() + 30
        time.sleep(3)

health = get_json(f"{RETRIEVAL_URL}/health")
print(json.dumps(health, indent=2))
assert health["collection_count"] == 131930, health
assert health["catalog"]["perfumes"] == 131930, health
assert health["catalog"]["similarity_edges"] > 600000, health
print("Retrieval service preflight: passed")
'''
    ),
    markdown("## Contract and quality smoke tests"),
    code(
        r'''
def post_json(path, payload, *, timeout=180):
    request = urllib.request.Request(
        f"{RETRIEVAL_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Retrieval HTTP {exc.code}: {body}") from exc


resolved = post_json("/resolve", {"hint": "Club de Nuit by Armaf"})
print("Resolved short name:", resolved["resolved"]["label"])
assert resolved["resolved"]["label"] == "Club de Nuit Intense Man by Armaf"

identity_contracts = {
    "YSL Y EDP": "Y Eau de Parfum by Yves Saint Laurent",
    "Y EPD by YSL": "Y Eau de Parfum by Yves Saint Laurent",
    "Bleu de Chanel EDP": "Bleu de Chanel Eau de Parfum by Chanel",
    "Bleu de Chanel EDT": "Bleu de Chanel by Chanel",
    "JPG Le Male Le Parfum": "Le Male Le Parfum by Jean Paul Gaultier",
    "LV Imagination": "Imagination by Louis Vuitton",
    "MFK Grand Soir": "Grand Soir by Maison Francis Kurkdjian",
    "PDM Layton": "Layton by Parfums de Marly",
}
for hint, expected_label in identity_contracts.items():
    identity = post_json("/resolve", {"hint": hint})["resolved"]
    assert identity and identity["label"] == expected_label, {"hint": hint, "resolved": identity}
print("Alias/edition identity contracts:", list(identity_contracts))

versace = post_json(
    "/search",
    {
        "query": "Recommend men's fragrances from Versace",
        "top_k": 8,
        "filters": {"brand": "Versace", "gender": "male"},
    },
)
assert len(versace["results"]) >= 5, versace
assert all(item["brand"] == "Versace" for item in versace["results"]), versace
print("Versace brand filter:", [item["label"] for item in versace["results"][:5]])

office = post_json(
    "/search",
    {
        "query": "I need a clean office scent without vanilla.",
        "top_k": 10,
        "filters": {"time": "day", "min_popularity": 100},
        "wanted_terms": ["clean", "fresh", "soapy", "powdery"],
        "exclude_terms": ["vanilla"],
    },
)
assert office["results"], office
for item in office["results"]:
    traits = (item["metadata"].get("accords_csv", "") + "," + item["metadata"].get("notes_csv", "")).lower()
    assert "vanilla" not in traits, item
assert sum(item["brand"] == "Clean" for item in office["results"]) <= 1, office
print("Office/no-vanilla:", [item["label"] for item in office["results"][:5]])

similar = post_json("/similar", {"hint": "Aventus", "top_k": 10})
assert similar["source"]["label"] == "Aventus by Creed", similar
assert any(item["label"] == "Club de Nuit Intense Man by Armaf" for item in similar["results"][:5]), similar
print("Aventus graph matches:", [item["label"] for item in similar["results"][:5]])

print("\\nCLEAN RETRIEVAL CONTRACT TEST: PASSED")
'''
    ),
    markdown("## Warm latency benchmark and durable report"),
    code(
        r'''
import statistics
from datetime import datetime, timezone

benchmark_cases = [
    {"query": "fresh citrus summer cologne for men", "filters": {"gender": "male", "season": "summer"}, "wanted_terms": ["citrus", "fresh"]},
    {"query": "warm spicy fragrance for a winter date night", "filters": {"season": "winter", "time": "night"}, "wanted_terms": ["warm spicy"]},
    {"query": "clean office scent without vanilla", "filters": {"time": "day"}, "exclude_terms": ["vanilla"], "wanted_terms": ["clean", "soapy"]},
    {"query": "pineapple woody fragrance", "wanted_terms": ["pineapple", "woody"]},
    {"query": "old school masculine fougere", "filters": {"gender": "male"}, "wanted_terms": ["fougere", "aromatic"]},
]

rounds = []
for round_number in (1, 2):
    for case in benchmark_cases:
        wall_started = time.perf_counter()
        result = post_json("/search", {**case, "top_k": 10})
        rounds.append({
            "round": round_number,
            "query": case["query"],
            "wall_seconds": round(time.perf_counter() - wall_started, 4),
            "service_seconds": result["elapsed_seconds"],
            "embedding_cache_hit": result["embedding_cache_hit"],
            "top_labels": [item["label"] for item in result["results"][:5]],
        })
        print(round_number, case["query"], rounds[-1]["wall_seconds"], "s", "cache=", result["embedding_cache_hit"])

warm = [row["wall_seconds"] for row in rounds if row["round"] == 2]
assert all(row["embedding_cache_hit"] for row in rounds if row["round"] == 2), rounds
report = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "clean_retrieval_stage2",
    "health": get_json(f"{RETRIEVAL_URL}/health"),
    "copy_elapsed_seconds": round(copy_elapsed, 4),
    "warm_mean_seconds": round(statistics.mean(warm), 4),
    "warm_median_seconds": round(statistics.median(warm), 4),
    "runs": rounds,
    "contract_tests": {
        "canonical_family_resolution": True,
        "alias_and_edition_resolution": True,
        "brand_filter": True,
        "negative_filter": True,
        "community_similarity": True,
    },
}
REPORT_PATH = PROJECT_DIR / "runs" / "clean_retrieval_stage2_report.json"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))
print("Saved:", REPORT_PATH)
'''
    ),
    markdown(
        """
## Interactive structured retrieval

At this stage the caller supplies structured constraints explicitly. In Stage 3,
the already validated model planner will produce this same JSON contract.
"""
    ),
    code(
        r'''
my_request = {
    "query": "an elegant woody fragrance for autumn evenings, but no oud",
    "top_k": 10,
    "filters": {"season": "autumn", "time": "night"},
    "wanted_terms": ["woody"],
    "exclude_terms": ["oud"],
}
my_results = post_json("/search", my_request)
for index, item in enumerate(my_results["results"], 1):
    print(f"{index:02d}. {item['label']} | score={item['score']:.3f}")
print("Elapsed:", my_results["elapsed_seconds"], "seconds")
'''
    ),
]

notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "accelerator": "CPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT} with {len(cells)} cells")
