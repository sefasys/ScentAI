# ScentAI Clean Start

This directory is intentionally independent from the previous combined Colab
pipeline. Its first milestone is only:

`Gemma 4 12B + existing pilot LoRA -> successful local vLLM API response`

## Stage 1: inference only

Open `colab_scentai_inference_clean.ipynb` in a fresh A100 Colab runtime and use
**Run all**. The notebook:

1. Installs only the dependency-free `uv` launcher in the Colab kernel.
2. Creates `/content/scentai_inference_env` with managed Python 3.12.
3. Installs `vllm==0.25.1` and the `ninja` launcher required by FlashInfer JIT;
   vLLM chooses its compatible PyTorch backend.
4. Checks the isolated dependency graph and CUDA import path.
5. Validates the existing rank-16 standard LoRA on Drive.
6. Starts one localhost-only vLLM server with the base model and LoRA.
7. Runs one base-model request and one grounded LoRA request.

No retrieval or training library is installed in this notebook. In particular,
there is no Chroma, BGE, sentence-transformers, SciPy, sklearn, torchvision,
Unsloth, TRL, PEFT, or bitsandbytes dependency.

## Required Drive asset

The notebook expects:

`MyDrive/Perfume-Dataset/models/scentai-gemma-4-12b-it-pilot-fastmodel-lora/best_lora_adapter`

An optional Colab secret named `HF_TOKEN` may be provided. The current base model
repository is public, so the notebook can also try without it.

## Resume the first failed run

The first clean run reached model loading but exposed one missing FlashInfer JIT
tool: `ninja`. The notebook now installs it and prepends the isolated environment's
`bin` directory to the vLLM subprocess PATH.

If that same Colab runtime is still alive, it can be repaired without downloading
the base model again. Run this temporary cell once, then rerun **Start the local
vLLM server** and every cell below it:

```python
subprocess.run(
    [UV, "pip", "install", "--python", str(INFERENCE_PYTHON), "ninja"],
    check=True,
)
os.environ["PATH"] = str(INFERENCE_ENV / "bin") + os.pathsep + os.environ.get("PATH", "")
print(subprocess.check_output([str(INFERENCE_ENV / "bin" / "ninja"), "--version"], text=True))
```

## Stage boundary

Do not add retrieval packages to the inference notebook after it passes. Stage 2
runs retrieval in a second isolated environment/service. Stage 3 will connect both
services over localhost. That separation is a permanent architecture boundary,
not a temporary workaround.

## Stage 2: retrieval only

Open `colab_scentai_retrieval_clean.ipynb` in a separate fresh Colab runtime and
use **Run all**. A GPU is not required. The notebook:

1. Creates `/content/scentai_retrieval_env` with managed Python 3.12.
2. Installs only `chromadb==1.5.9` and `sentence-transformers==5.6.0` with
   CPU-only PyTorch.
3. Copies the Chroma index and SQLite catalog from Drive to `/content` for lower
   query latency.
4. Starts a localhost-only retrieval HTTP service on port 8020.
5. Tests canonical family resolution, strict brand filtering, negative filtering,
   brand diversity, and community-graph similarity.
6. Runs a two-round latency benchmark and saves
   `runs/clean_retrieval_stage2_report.json` to Drive.

Semantic search retrieves a 300-item ANN candidate pool before reranking. The
reranker combines BGE-M3 relevance with explicit wanted traits, confidence-aware
ratings, popularity evidence, hard negative filtering, and brand diversity. This
larger pool costs little after warm-up but avoids losing strong candidates before
the structured constraints are applied.

The service API is intentionally structured:

- `POST /search`: BGE-M3 semantic retrieval plus metadata and negative filters.
- `POST /resolve`: canonical perfume identity resolution.
- `POST /similar`: Fragrantica community graph plus structural similarity.
- `GET /health`: collection, catalog, cache, and uptime checks.

Stage 2 does not start or import vLLM. Stage 3 will launch the two already proven
services together and connect them through localhost without merging their Python
dependencies.

## Stage 3: full grounded pipeline

Open `colab_scentai_stage3_pipeline.ipynb` in a fresh A100 High-RAM Colab
runtime and use **Run all**. This notebook keeps the proven Stage 1 and Stage 2
environments isolated, starts both localhost services, and adds a dependency-free
orchestrator in the Colab kernel.

The complete request path is:

`free-form query -> base Gemma planner -> retrieval route -> ScentAI LoRA -> validator`

The planner is model-based rather than a fixed keyword router. Deterministic code
is limited to evidence validation, exact database lookup, hard exclusions,
canonical identity resolution, and output safety. Invalid generated answers are
retried once and then replaced with a grounded fallback.

Stage 3 writes:

- `runs/stage3_pipeline_report.json`: end-to-end contract results.
- `runs/stage3_interactive_results.jsonl`: durable interactive query history.

See `STAGE3_COLAB_CALISTIRMA.md` for the exact Drive layout and run sequence.
