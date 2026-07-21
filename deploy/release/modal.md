# Stage 6: Modal Deployment

Stage 6 deploys the frozen V4 pipeline without changing its model behavior:

- one private A100-80GB vLLM worker for Gemma 4 12B BF16 and the rank-16 LoRA;
- one private CPU retrieval worker for BGE-M3, Chroma, and SQLite;
- one public FastAPI endpoint protected by `X-API-Key`;
- scale-to-zero and a hard maximum of one GPU container;
- CPU-only artifact validation and Hugging Face prefetch;
- 12-case smoke and 120-case cloud regression runners.

The base model still plans and writes the first answer. The `scentai` LoRA remains the repair
model. This is intentional and matches the 120/120 V4 release.

## 1. One-time local setup

These commands upload only source code and command metadata. They do not download the 24 GB base
model to the local machine.

```bash
python -m pip install -r deploy/requirements-modal.txt
modal setup
python deploy/scripts/modal_bootstrap.py
```

The bootstrap creates:

- Volumes: `scentai-models`, `scentai-data`, `scentai-hf-cache`, `scentai-vllm-cache`
- Secrets: `scentai-huggingface`, `scentai-api`

To allow a separate browser frontend, pass its exact origin:

```bash
python deploy/scripts/modal_bootstrap.py --cors-origin https://your-frontend.example
```

Do not put Modal, Hugging Face, or ScentAI API tokens in the repository.

## 2. Upload artifacts from Google Drive

Open `notebooks/upload_modal_artifacts_colab.ipynb` directly in Colab and run its cells from top
to bottom. The notebook mounts Drive, securely prompts for the Modal API token pair, validates the
full adapter/Chroma/catalog, and uploads all three artifacts.

The notebook prompts for a Modal token ID and secret without writing them into the notebook. Its
default Drive layout is:

```text
/content/drive/MyDrive/Perfume-Dataset/
├── chroma_db_bge_m3/
├── scentai_catalog.sqlite3
└── models/
    └── scentai-gemma-4-12b-it-full-fastmodel-lora/
        └── best_lora_adapter/
```

It uploads the directory contents to these exact Volume paths:

```text
/models/scentai/adapter_config.json
/models/scentai/adapter_model.safetensors
/data/chroma_db_bge_m3/chroma.sqlite3
/data/scentai_catalog.sqlite3
```

If the full adapter is elsewhere, change only `ADAPTER_DIR` in the configuration cell. Do not
upload the pilot adapter. The standalone `modal_colab_upload.py` remains available for Colab
sessions that support uploaded Python files.

## 3. Validate artifacts

This runs on CPU and does not start or bill an A100:

```bash
modal run deploy/modal_setup_app.py::preflight
```

It rejects a missing artifact, wrong base model, wrong rank, DoRA adapter, or changed target
modules before vLLM can start.

## 4. Prefetch model snapshots

This also runs on CPU. Hugging Face downloads directly into a Modal Volume, so the base model and
BGE-M3 never pass through the local internet connection:

```bash
modal run deploy/modal_setup_app.py::prefetch
```

Run it once. Later containers reuse `scentai-hf-cache` and vLLM's compilation cache.

## 5. Deploy

```bash
modal deploy deploy/modal_app.py --tag scentai-v1.0-rc2
```

Modal prints the public `web` URL. Store it as `SCENTAI_MODAL_URL` without a trailing slash.

`GET /health/live` checks only the public CPU process. `GET /health/ready`, runtime warm-up jobs,
`POST /v1/chat`, and the regression runner start the GPU when it is cold.

## 6. Public API smoke test

```bash
export SCENTAI_MODAL_URL='https://YOUR-WORKSPACE--scentai-stage6-web.modal.run'
read -s SCENTAI_API_KEY && export SCENTAI_API_KEY
python deploy/scripts/modal_http_smoke.py --url "$SCENTAI_MODAL_URL"
```

Alternatively, keep the API key entirely inside Modal and run the same four cases through a
temporary CPU smoke function:

```bash
modal run deploy/modal_app.py::public_smoke \
  --url "$SCENTAI_MODAL_URL" \
  --output deploy/reports/modal_http_smoke.json
```

The first call includes vLLM cold start and is not a latency measurement. The report is written
to `deploy/reports/modal_http_smoke.json`.

## 7. Cloud regression gates

Run the 12-case cross-category smoke first:

```bash
modal run deploy/modal_app.py::regression \
  --limit 12 \
  --output deploy/reports/modal_regression_smoke.json
```

Only after all 12 pass, run the frozen 120 cases:

```bash
modal run deploy/modal_app.py::regression \
  --limit 0 \
  --output deploy/reports/modal_regression_full.json
```

`--limit 0` means all cases. A regression failure exits non-zero and keeps the complete output
for diagnosis.

## Cost and lifecycle controls

- `min_containers=0`: no always-on GPU.
- `max_containers=1`: no accidental second A100.
- GPU `scaledown_window=300`: the A100 stops after roughly five idle minutes.
- API and retrieval workers stay warm for up to 20 idle minutes.
- Hugging Face prefetch and artifact preflight are CPU-only.
- The GPU uses `gpu-memory-utilization=0.65`; this changes KV-cache capacity, not model weights or
  answer quality.

A conversation remains available while the single API container is alive. A redeploy or API
scale-down clears in-memory sessions; persistent multi-replica sessions belong to Stage 7.

The Stage 7A local frontend origins (`http://127.0.0.1:5173` and
`http://localhost:5173`) are explicitly allowed by the public API. Replace or extend this list
with the final frontend origin before its production deployment.

Browser clients use `POST /v1/chat/jobs` followed by `GET /v1/chat/jobs/{job_id}`. This avoids
Modal's 150-second Web Function HTTP redirect boundary, which cannot carry application CORS
headers when a GPU cold start outlives the original cross-origin request. Server-to-server API
clients may continue to use the synchronous `POST /v1/chat` contract.

Before showing the composer, the Stage 7A frontend uses `POST /v1/runtime/warmup/jobs` followed by
`GET /v1/runtime/warmup/jobs/{job_id}`. The warm-up screen owns the cold-start wait, and the chat UI
appears only after both retrieval and the base/LoRA model endpoints report ready.

## Expected cold and warm behavior

The first request after scale-to-zero can take several minutes while vLLM maps Gemma 4 into GPU
memory. Warm requests should be compared with the frozen V4 latency report, not with cold start.
If the five-minute window is too aggressive during demos, raise only `ModelWorker.scaledown_window`
in `deploy/modal_app.py`; do not raise `min_containers` unless always-on billing is intended.
