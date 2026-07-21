# Quality-First Runtime Optimization Plan

## Objective

Optimization must reduce wasted work, not remove recommendation evidence. Hard constraints remain deterministic; soft preferences remain multi-signal and rankable.

The runtime is a constrained multi-objective system:

- Eligibility: exclusions, brand, gender, year, owned bottles, and exact result count.
- Relevance: semantic intent, accords, notes, season, and time.
- Community evidence: smell similarity and taste affinity.
- Confidence: vote volume, Bayesian rating, and edge reliability.
- Diversity: brand, product-family, and profile coverage.
- Explanation quality: grounded Gemma response.

Eligibility is a hard gate. The remaining objectives are fused and reranked; they must not be turned into one growing list of hardcoded rules.

## Measured Baseline

Local CPU measurements:

| Operation | Measured cost |
|---|---:|
| Load BGE-M3 + open Chroma | 5.90 s, once per process |
| Full BGE/Chroma retrieval query | 0.46-1.45 s |
| Mean retrieval query | 0.94 s |
| Repeated identical retrieval after embedding cache | 0.43 s (from 1.29 s cold) |
| 1,000 indexed catalog records | 7.6 ms |
| 500 graph-neighbor lookups | 45 ms cold |
| Five exact identity resolutions after label index | 0.6 ms cold |
| 5,000 cached identity resolutions | 0.3 ms |
| 5,000 cached graph lookups | 0.7 ms |

Storage:

- Chroma/BGE-M3 index: 1.5 GB.
- Current SQLite catalog: 94 MB.
- Compressed catalog: 33 MB.

The graph and deterministic catalog are not the bottleneck. Repeated embedding and Gemma generation dominate latency.

## Cascaded Execution

### Route 0: model-planned, no second generation

Gemma remains the primary intent planner. Exact lookup and unsupported live-data requests can skip the second answer-generation call after planning. Repeated normalized queries reuse a bounded planner cache. The deterministic parser is only a failure/safety fallback.

### Route 1: graph-first reference retrieval

For `smells like`, `dupe`, or `if I like` queries:

1. Resolve the source identity.
2. Read high-confidence graph neighbors.
3. Apply hard filters.
4. If enough diverse candidates remain, exactly rescore those candidates and skip broad ANN search.
5. Fall back to BGE-M3 only when graph coverage or confidence is insufficient.

This is conditional computation, not a quality reduction. BGE remains the fallback safety net.

### Route 2: semantic retrieval

Use one cached BGE embedding and metadata pushdown for free-form vibe, occasion, or messy queries. Retrieve a broad pool, then exactly rescore 80-200 candidates using structured and community signals.

### Route 3: expensive ambiguity path

Use multi-channel retrieval and an optional cross-encoder only for ambiguous, multi-reference, or low-confidence cases. Rerank 30-50 candidates, not the full catalog.

### Final generation

Send only 6-10 diverse final cards to Gemma. Deterministic routes bypass Gemma; generated routes retain validation, one retry, and grounded fallback.

## Quality-Preserving Optimizations

### Precompute offline

- Canonical duplicate IDs and merged graph evidence.
- Bayesian rating and edge-confidence scores.
- Weighted accord vectors and layer-aware note vectors.
- Top graph neighbors per intent type.
- Optional graph embeddings or personalized-PageRank seeds.

Precomputation replaces repeated arithmetic; it does not discard signals.

### Cache safely

- Identity resolution: LRU, already implemented.
- Graph neighborhoods: LRU, already implemented.
- Query embeddings: bounded LRU, already implemented.
- Retrieval result cache: key by normalized query, hard filters, model version, and catalog version.

Never cache only the raw text while ignoring user filters or catalog/model version.

### Preserve candidate recall

- ANN search is candidate generation, not the final decision.
- Keep broad `fetch_k` for difficult queries.
- Apply exact scores after ANN retrieval.
- Skip BGE only when a graph route passes minimum count, confidence, and diversity gates.

### Fuse instead of overconstrain

Use calibrated reciprocal-rank fusion or a small learned reranker. Keep separate features for semantic relevance, smell graph, taste graph, structure, quality, and popularity. A missing edge is unknown, not negative.

### Control LLM cost

- Dynamic card count based on requested result count.
- Compact cards containing only fields relevant to the query.
- Dynamic generation length.
- No second generation when the first response passes validation.
- Deterministic template fallback after one failed retry.

## Training Optimization

Do not force all database logic into the LoRA. That increases memorization pressure and becomes stale when the catalog changes.

- Fine-tune Gemma for instruction following, grounded explanation, and refusal behavior.
- Train a smaller reranker on graph-derived positive/hard-negative pairs for ranking quality.
- Keep live catalog facts and hard constraints in retrieval/runtime.
- Distinguish smell-similarity supervision from taste-affinity supervision.

This division improves both optimization stability and updateability.

## Quality Gates

Every optimization must pass the same frozen evaluation before release:

- Hard-filter violations: 0.
- Exact entity resolution regression: 0.
- Retrieval intent pass rate: no decrease.
- Graph holdout Recall@K and NDCG@K: no decrease.
- Duplicate rate in final lists: 0.
- Catalog coverage and brand diversity: no material decrease.
- Grounding/hallucination pass rate: no decrease.
- Latency and memory measured only after quality gates pass.

An optimization that is faster but fails a quality gate is rejected.

## Performance Claim Calibration

Runtime calibration and narrow contradiction validation are implemented in the Stage 3
orchestrator. Broader social-outcome claim evaluation remains part of final evaluation.

- Calibrate longevity and sillage language against the catalog distribution instead of
  relying on the model's ungrounded labels such as "low", "medium", or "strong".
- Keep evidence-backed sensory interpretation available. Phrases such as "evokes" or
  "is reminiscent of" may infer a vibe from listed notes and accords, but must not turn
  that inference into a claim that an unlisted note is present.
- Avoid guaranteed social or room-performance claims. When performance evidence is
  missing, describe it as unknown; when present, use calibrated and non-absolute wording.
- Current calibration reference: Turkish Leather by Pryn Parfum has longevity 4.32/5
  (about the 95th catalog percentile) and sillage 2.79/4 (about the 85th percentile), so
  describing its longevity as "medium" is incorrect.

## Implementation Order

1. Keep the new label index, Bayesian rating, deduplication, and bounded caches.
2. Build catalog v2 with canonical IDs, weighted accords, note layers, and `also_liked` edges.
3. Add graph-first routes with explicit confidence/fallback gates.
4. Add intent-aware rank fusion and diversity reranking.
5. Build graph holdout evaluation before tuning weights or training a reranker.
6. Optimize Gemma context and generation only after retrieval quality is frozen.
