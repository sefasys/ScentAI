# ScentAI Deployment V1

This directory turns the validated V4 notebook pipeline into three deployable services without
changing model behavior:

1. **Release freeze**: hashes the V4 source, evaluation, catalog, and Chroma snapshot.
2. **Runtime package**: imports the tested `scentai` engine without notebook globals.
3. **Public API**: FastAPI chat, health, API-key protection, and bounded multi-turn sessions.
4. **Model server**: vLLM 0.25.1 serving Gemma 4 12B BF16 plus the full LoRA under `scentai`.
5. **Retrieval service**: one CPU BGE-M3/Chroma/SQLite process shared by all API requests.
6. **Modal target**: scale-to-zero A100 inference with private model/retrieval workers and a
   public API-key-protected ASGI endpoint.

The frozen routing remains intentional: the base model plans and writes the first answer; the
LoRA is used only for a failed-answer repair. This is the V4 configuration that passed 120/120
functional evaluation cases.

## Verify the release

```bash
python deploy/scripts/verify_release.py
```

When the full adapter is available locally, include it in verification:

```bash
python deploy/scripts/verify_release.py \
  --adapter /absolute/path/to/best_lora_adapter
```

## Configure and run

```bash
cp deploy/.env.example deploy/.env
# Edit deploy/.env, especially SCENTAI_ADAPTER_HOST_DIR and SCENTAI_API_KEY.
docker compose --env-file deploy/.env -f deploy/compose.yaml config
docker compose --env-file deploy/.env -f deploy/compose.yaml up --build -d
docker compose --env-file deploy/.env -f deploy/compose.yaml ps
```

Only the API is published to the host. vLLM and retrieval stay on the internal Compose network.
The first model/BGE-M3 download is cached in `SCENTAI_HF_CACHE_DIR`.

```bash
curl -s http://localhost:8080/health/ready
curl -s http://localhost:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: replace-with-a-long-random-secret' \
  -d '{"query":"Ofis için temiz ve vanilyasız üç parfüm öner."}'
```

Send the returned `session_id` on later turns. Unknown or expired IDs return 404 instead of
silently starting an unrelated conversation.

## GPU memory

The Colab notebook used `gpu-memory-utilization=0.86`, which made vLLM reserve roughly 69 GB on
an 80 GB A100. Deployment defaults to `0.65` (roughly 52 GB). This does not quantize or replace
Gemma 4; it reduces KV-cache capacity, so quality is unchanged while maximum concurrency falls.
Raise the value only after load testing shows that concurrent traffic needs a larger cache.

## Tests

```bash
python -m pip install -r deploy/requirements-test.txt
PYTHONPATH="$PWD/deploy/src:$PWD" pytest -q \
  deploy/tests tests/test_entity_resolution.py tests/test_orchestrator_contracts.py

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python deploy/scripts/smoke_retrieval.py

python deploy/scripts/http_smoke_retrieval.py
```

The API intentionally uses one worker in Stage 5 because session state is in memory. Redis-backed
sessions and horizontal API replicas belong to the next deployment stage.

The retrieval process also owns one dedicated worker thread. BGE-M3 is constructed and queried on
that same thread, avoiding PyTorch thread-affinity stalls while its 512-entry embedding cache is
shared across every request.

## Modal deployment

The Docker Compose target remains useful for a reserved GPU host. For the selected on-demand
deployment, follow [modal.md](release/modal.md). It includes Drive-to-Modal artifact
upload, CPU-only model prefetch, deploy commands, public API smoke tests, and the frozen 120-case
cloud regression gate.
