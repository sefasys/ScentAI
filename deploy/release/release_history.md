# Stages 1-5 Completion Record

## 1. Frozen release

- Release: `scentai-v1.0-rc1`
- Manifest: `scentai-v1.0-rc1.manifest.json`
- V4 evaluation: 120/120 functional cases passed.
- Catalog integrity: 131,930 perfumes and 692,729 similarity edges.
- Full adapter remains an external required artifact and is validated before model startup.

## 2. Production runtime package

- Environment-driven configuration under `deploy/src/scentai_deploy`.
- The tested V4 orchestrator remains the canonical behavior source.
- Multi-turn state is bounded, expiring, thread-safe, and isolated by UUID session.

## 3. FastAPI backend

- `POST /v1/chat`
- `DELETE /v1/sessions/{session_id}`
- `/health/live` and dependency-aware `/health/ready`
- Optional API key and explicit CORS allowlist.
- One API process with a managed request executor; no model duplication per web worker.

## 4. vLLM model service

- Frozen image: `vllm/vllm-openai:v0.25.1-x86_64-cu129`.
- Base model, first answer: `google/gemma-4-12B-it`.
- Repair model: dynamic rank-16 LoRA served as `scentai`.
- BF16 and 4096-token context preserve V4 behavior.
- Default GPU reservation lowered from 0.86 to 0.65 for an A100 80 GB deployment.

## 5. Retrieval service

- BGE-M3, Chroma, and SQLite load once in a CPU service.
- Legacy `/health`, `/search`, `/resolve`, and `/similar` contracts are preserved.
- BGE-M3 is constructed and queried on one dedicated thread.
- Real snapshot smoke test passed for health, alias resolution, search, and vanilla exclusion.

