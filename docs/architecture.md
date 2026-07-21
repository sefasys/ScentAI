# Architecture

## Request Lifecycle

1. The client starts an asynchronous warm-up job before opening the composer.
2. A model-based planner converts the free-form request into an evidence-bearing JSON plan.
3. Planner constraints are checked against the exact user text.
4. Retrieval resolves named perfumes or performs BGE-M3 semantic search.
5. SQLite metadata, community similarity edges, ratings, popularity, and hard filters rerank the candidate pool.
6. Grounded perfume cards are sent to Gemma 4.
7. The answer validator checks catalog identity, unsupported facts, requested counts, hard exclusions, language, and performance calibration.
8. Invalid answers receive one repair attempt through the LoRA. A deterministic grounded fallback is used only when both generations fail validation.

## Runtime Boundaries

The system keeps three environments separate:

- **Model worker:** vLLM, CUDA, Gemma 4, and the dynamic LoRA.
- **Retrieval worker:** CPU PyTorch, BGE-M3, Chroma, and SQLite.
- **API worker:** orchestration, sessions, validation, and asynchronous jobs.

This separation was introduced after combined notebook environments repeatedly produced incompatible CUDA, Transformers, Pillow, torchvision, and vLLM dependency graphs.

## Retrieval

Retrieval starts with a broad semantic pool and then applies structured scoring. Important signals include:

- semantic similarity;
- required and preferred traits;
- hard negative filters;
- confidence-adjusted rating and vote count;
- mainstream or niche discovery mode;
- brand diversity;
- canonical name confidence;
- community similarity votes for reference-fragrance queries.

Product-name matching is catalog-wide. It does not contain a one-off alias for a single perfume family. Exact names, shortened family names, brand-qualified names, and common abbreviations are resolved through normalized catalog evidence and dominance signals.

## Generation Policy

The model is responsible for semantic understanding and consultant-style prose. Code remains deterministic where a language model should not be trusted:

- copying exact database fields;
- applying explicit exclusions;
- verifying named products;
- enforcing counts;
- rejecting unsupported live or medical claims;
- preventing fabricated catalog facts.

The production release uses the base model for planning and first-answer generation. The fine-tuned adapter is a validation-triggered repair model. This configuration performed better than forcing every request through the adapter.

## Deployment Path

```text
Browser
  -> Cloudflare Worker and static assets
  -> Modal public FastAPI worker
  -> private Modal model worker
  -> private Modal retrieval worker
```

The browser never receives the Modal API key. Cloudflare injects it server-side and allowlists the asynchronous warm-up, chat, polling, and session-deletion routes.

