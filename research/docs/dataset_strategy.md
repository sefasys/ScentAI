# Dataset Size Plan

## Goal

Build a balanced ScentAI training set where each level teaches a distinct behavior.

The dataset should not be dominated by cheap template records. L1 and L2 are important for factual discipline and filtering, but the product value comes from L3, L4, and later L5.

## Level Roles

```text
L1: factual perfume knowledge
L2: strict filtering and comparison
L3: semantic recommendation, list-only
L4: grounded recommendation with short reasoning
L5: preference/profile-aware personalization
```

## Main Recommendation

Ideal target:

```text
Total: 32,000 records

L1: 4,000    12.5%
L2: 6,000    18.75%
L3: 7,000    21.875%
L4: 10,000   31.25%
L5: 5,000    15.625%
```

Why this balance:

- L1 is intentionally limited. It teaches factual discipline, but too much L1 would make the model feel like a database bot.
- L2 is larger than L1 because strict filtering has many combinations, negatives, rankings, comparisons, and no-match cases.
- L3 is one of the core layers because real users ask vague semantic questions, not clean database filters.
- L4 is the largest layer because the target product needs grounded recommendations with convincing short explanations.
- L5 is substantial but not dominant because preference memory is important, yet easier to overfit if the profile format is too repetitive.

## Practical Production Path

Do not produce the full 25K immediately.

### Stage 1: QA Batch

```text
Total: 2,000 records

L1: 250
L2: 350
L3: 450
L4: 700
L5: 250
```

Purpose:

- validate schemas
- inspect answer tone
- check RAG subset behavior
- check query diversity
- catch category-specific failure modes

### Stage 2: First Training Run

```text
Total: 12,000 records

L1: 1,500
L2: 2,250
L3: 2,750
L4: 3,750
L5: 1,750
```

Purpose:

- train first meaningful model
- evaluate whether the model overfits template behavior
- test L3/L4 quality before spending many API days

### Stage 3: Main Dataset

```text
Total: 32,000 records

L1: 4,000
L2: 6,000
L3: 7,000
L4: 10,000
L5: 5,000
```

Purpose:

- main production-quality training set
- enough reasoning and personalization data without letting simple templates dominate

## Alternative Smaller Plan

If API quotas become painful:

```text
Total: 20,000 records

L1: 2,500
L2: 4,000
L3: 4,500
L4: 6,000
L5: 3,000
```

This keeps L4 strong while reducing total generation time.

## Alternative Larger Plan

If generation quality is good and quota/cost is acceptable:

```text
Total: 50,000 records

L1: 6,000
L2: 9,000
L3: 11,000
L4: 16,000
L5: 8,000
```

Do this only after the 12K run proves useful and L5 is stable.

## Production Order

Recommended order:

1. Finish L1 and L2 generation because they are deterministic and cheap.
2. Produce L3 in chunks with `--fallback-policy fail`.
3. Produce L4 in chunks with checkpoint/resume.
4. Design L5 after L4 quality is accepted.
5. Merge only records that pass validators and manual spot checks.

## API/Quota Notes

L1 and L2 are programmatic and should not require LLM API calls.

L3 and L4 use the LLM mainly for user query generation, so produce them in daily chunks.

For Groq free/on-demand limits:

- keep checkpoint interval at 100
- use `--resume` after rate limits
- keep `--fallback-policy fail` for final data
- prefer several clean 500-record chunks over one fragile large run

## Current Decision

Use the 32K plan as the main target unless later training results show overfitting or quality imbalance.

Temporary working target before L5 is ready:

```text
L1: 4,000
L2: 6,000
L3: 7,000
L4: 10,000
L5: pending
```

When L5 is implemented, add the first 5,000 profile-aware records.
